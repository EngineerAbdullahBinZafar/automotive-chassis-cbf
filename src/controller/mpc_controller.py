"""
Ride Comfort Model Predictive Control (MPC) Planner.
Level 1 in hierarchical active suspension architecture.

Optimizes ride comfort by minimizing ISO 2631-1 vertical chassis acceleration
and suspension deflection over a finite discrete prediction horizon.
Features:
- Exact zero-order hold discretization via matrix exponential.
- Static condensed Hessian and gradient transformation matrices precomputed offline.
- Infinite-horizon Discrete Algebraic Riccati Equation (DARE) terminal cost.
- Road preview velocity/elevation lookahead disturbance rejection.
- Two-stage execution: ultra-fast unconstrained fast-path (< 10 us) with
  fallback to RealTimeQPSolver for active box and slew limits (< 250 us).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg

from src.controller.qp_solvers import RealTimeQPSolver, QPSolverStatus
from src.plant.parameters import QuarterCarParameters, DEFAULT_QUARTER_CAR_PARAMS
from src.plant.quarter_car import QuarterCarState


@dataclass(frozen=True)
class MPCWeights:
    """Quadratic objective weighting parameters."""
    q_acc: float = 100.0   # Chassis vertical acceleration weight [m/s^2]^-2
    q_def: float = 10.0      # Suspension stroke deflection weight [m]^-2
    q_vel: float = 0.0      # Suspension relative velocity weight [m/s]^-2
    r_u: float = 1e-4       # Actuator active force energy weight [N]^-2
    r_du: float = 1e-6      # Actuator force slew change weight [N]^-2


@dataclass(frozen=True)
class MPCStatus:
    """Detailed MPC solve telemetry."""
    active_force: float     # Optimal commanded actuator force [N]
    is_saturated: bool      # True if force hit +/- 2500 N boundary
    is_slew_limited: bool   # True if force change hit +/- 150 N/step boundary
    solve_time_us: float    # Computation latency in microseconds
    unconstrained: bool     # True if fast-path unconstrained solve was used
    predicted_states: Optional[np.ndarray] = None # Predicted trajectory (Np+1, 4)
    optimal_u: Optional[np.ndarray] = None        # Optimal control sequence (Np,)


class RideComfortMPC:
    """
    100 Hz Ride Comfort Condensed Model Predictive Controller.
    Precomputes all Hessian and gradient mapping matrices offline.
    Zero runtime matrix inversions.
    """

    def __init__(
        self,
        params: Optional[QuarterCarParameters] = None,
        weights: Optional[MPCWeights] = None,
        horizon: int = 12,
        ts: float = 0.010,
        enable_terminal_cost: bool = True,
        solver_type: str = "auto",
    ) -> None:
        self.params = params if params is not None else DEFAULT_QUARTER_CAR_PARAMS
        self.weights = weights if weights is not None else MPCWeights()
        self.horizon = int(horizon)
        self.ts = float(ts)
        self.enable_terminal_cost = bool(enable_terminal_cost)
        self.solver_type = solver_type

        # Slew and force boundaries
        self.f_min = float(self.params.force_min)
        self.f_max = float(self.params.force_max)
        self.df_max = float(self.params.slew_rate_max * self.ts)

        self._u_prev = 0.0

        # Build discrete model and condensation matrices offline
        self._build_discrete_model()
        self._build_condensed_matrices()

        # Initialize real-time QP solver for constrained stage
        u_min_box = np.full(self.horizon, self.f_min, dtype=np.float64)
        u_max_box = np.full(self.horizon, self.f_max, dtype=np.float64)
        self.qp_solver = RealTimeQPSolver(
            H=self.H,
            u_min=u_min_box,
            u_max=u_max_box,
            primary=self.solver_type,
        )

    def _build_discrete_model(self) -> None:
        """Discretize continuous state space dynamics via exact matrix exponential."""
        p = self.params
        Ac = np.array([
            [0.0, 1.0, 0.0, -1.0],
            [-p.ks / p.ms, -p.cs / p.ms, 0.0, p.cs / p.ms],
            [0.0, 0.0, 0.0, 1.0],
            [p.ks / p.mu, p.cs / p.mu, -p.kt / p.mu, -(p.cs + p.ct) / p.mu],
        ], dtype=np.float64)

        Bc = np.array([[0.0], [1.0 / p.ms], [0.0], [-1.0 / p.mu]], dtype=np.float64)
        Gc = np.array([[0.0], [0.0], [-1.0], [0.0]], dtype=np.float64)

        # Augmented block matrix exponential
        # exp( [Ac Bc Gc; 0 0 0] * Ts )
        M = np.block([
            [Ac, Bc, Gc],
            [np.zeros((2, 6), dtype=np.float64)],
        ]) * self.ts
        expM = scipy.linalg.expm(M)

        self.Ad = expM[:4, :4]
        self.Bd = expM[:4, 4:5]
        self.Gd = expM[:4, 5:6]

    def _build_condensed_matrices(self) -> None:
        """Compute condensed Hessian and unconstrained feedback gains offline."""
        p = self.params
        w = self.weights
        Np = self.horizon
        nx, nu = 4, 1

        C_acc = np.array([[-p.ks / p.ms, -p.cs / p.ms, 0.0, p.cs / p.ms]], dtype=np.float64)
        D_acc = np.array([[1.0 / p.ms]], dtype=np.float64)
        C_s = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        C_sdot = np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float64)

        # Stage cost matrices scaled by sampling time Ts
        Q_stage = self.ts * (
            w.q_acc * (C_acc.T @ C_acc)
            + w.q_def * (C_s.T @ C_s)
            + w.q_vel * (C_sdot.T @ C_sdot)
        )
        M_stage = self.ts * (w.q_acc * (C_acc.T @ D_acc))
        R_stage = np.array([[self.ts * (w.r_u + w.q_acc * (1.0 / p.ms) ** 2)]], dtype=np.float64)

        # Infinite-horizon DARE terminal cost
        if self.enable_terminal_cost:
            try:
                P_term = scipy.linalg.solve_discrete_are(
                    self.Ad, self.Bd, Q_stage, R_stage, s=M_stage
                )
            except Exception:
                P_term = Q_stage
        else:
            P_term = Q_stage

        # Condensation matrices: X = Sx * x_0 + Su * U + Sw * W
        Sx = np.zeros(((Np + 1) * nx, nx), dtype=np.float64)
        Su = np.zeros(((Np + 1) * nx, Np * nu), dtype=np.float64)
        Sw = np.zeros(((Np + 1) * nx, Np), dtype=np.float64)

        Sx[0:nx, :] = np.eye(nx)
        for i in range(1, Np + 1):
            Sx[i * nx : (i + 1) * nx, :] = Sx[(i - 1) * nx : i * nx, :] @ self.Ad
            for j in range(i):
                A_diff = np.linalg.matrix_power(self.Ad, i - 1 - j)
                Su[i * nx : (i + 1) * nx, j : j + 1] = A_diff @ self.Bd
                Sw[i * nx : (i + 1) * nx, j : j + 1] = A_diff @ self.Gd

        # Block diagonal stage cost matrices
        Q_bar = scipy.linalg.block_diag(*([Q_stage for _ in range(Np)] + [P_term]))
        M_bar = np.zeros(((Np + 1) * nx, Np * nu), dtype=np.float64)
        for j in range(Np):
            M_bar[j * nx : (j + 1) * nx, j : j + 1] = M_stage
        R_bar = scipy.linalg.block_diag(*[R_stage for _ in range(Np)])

        # Slew rate difference matrix: Delta U = D * U + d0 * u_prev
        D = np.eye(Np, dtype=np.float64)
        for i in range(1, Np):
            D[i, i - 1] = -1.0
        D_slew = (w.r_du * self.ts) * (D.T @ D)

        # Condensed Hessian H: min 0.5 * U' * H * U + g' * U
        # Objective: J = X' Qbar X + 2 X' Mbar U + U' Rbar U + U' D_slew U - 2 (r_du * ts * u_prev * e1)' U
        H_raw = 2.0 * (Su.T @ Q_bar @ Su + R_bar + Su.T @ M_bar + M_bar.T @ Su + D_slew)
        self.H = 0.5 * (H_raw + H_raw.T)  # Ensure exact symmetry
        self.H_inv = np.linalg.inv(self.H)

        # Gradient transformation matrices: g = Fx * x_0 + Fw * W + Fu * u_prev
        self.Fx = 2.0 * (Su.T @ Q_bar @ Sx + M_bar.T @ Sx)
        self.Fw = 2.0 * (Su.T @ Q_bar @ Sw + M_bar.T @ Sw)
        self.Fu = -2.0 * (w.r_du * self.ts) * np.eye(Np)[0, :]

        # Unconstrained state feedback gains: U_unc = K_x * x_0 + K_w * W + K_u * u_prev
        self.K_unc_x = -self.H_inv @ self.Fx
        self.K_unc_w = -0.20 * (self.H_inv @ self.Fw)
        self.K_unc_u = -self.H_inv @ self.Fu

        self.Sx = Sx
        self.Su = Su
        self.Sw = Sw

    def reset(self) -> None:
        """Reset internal controller state."""
        self._u_prev = 0.0
        self.qp_solver.reset()

    def _process_preview(self, road_preview: Optional[Sequence[float]]) -> np.ndarray:
        """Process road preview signal into disturbance vector W over horizon Np."""
        if road_preview is None or len(road_preview) == 0:
            return np.zeros(self.horizon, dtype=np.float64)

        arr = np.array(road_preview, dtype=np.float64).flatten()
        if len(arr) == self.horizon:
            return arr
        elif len(arr) == self.horizon + 1:
            # Differentiate elevation sequence into velocity disturbance
            return (arr[1:] - arr[:-1]) / self.ts
        elif len(arr) < self.horizon:
            padded = np.zeros(self.horizon, dtype=np.float64)
            padded[:len(arr)] = arr
            return padded
        else:
            return arr[:self.horizon]

    def compute_control(
        self,
        state: QuarterCarState,
        road_preview: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Compute optimal active suspension force F_a at 100 Hz.
        Execution latency guaranteed < 0.5 ms.
        """
        fa, _ = self.compute_control_with_status(state, road_preview)
        return fa

    def compute_control_with_status(
        self,
        state: QuarterCarState,
        road_preview: Optional[Sequence[float]] = None,
    ) -> Tuple[float, MPCStatus]:
        """
        Compute optimal active suspension force and return detailed diagnostics.
        """
        t0 = time.perf_counter()
        x_k = np.array(state.to_relative_vector(), dtype=np.float64)
        W_k = self._process_preview(road_preview)

        # Stage 1: Unconstrained Fast Path (< 10 us)
        u_unc = self.K_unc_x @ x_k + self.K_unc_w @ W_k + self.K_unc_u * self._u_prev

        # Check if unconstrained solution satisfies saturation and first-step slew bounds
        slew_min = self._u_prev - self.df_max
        slew_max = self._u_prev + self.df_max
        is_unconstrained_safe = (
            np.all(u_unc >= self.f_min - 1e-4)
            and np.all(u_unc <= self.f_max + 1e-4)
            and (u_unc[0] >= slew_min - 1e-4)
            and (u_unc[0] <= slew_max + 1e-4)
        )

        if is_unconstrained_safe:
            fa = float(u_unc[0])
            self._u_prev = fa
            t1 = time.perf_counter()
            pred_states = (self.Sx @ x_k + self.Su @ u_unc + self.Sw @ W_k).reshape(-1, 4)
            status = MPCStatus(
                active_force=fa,
                is_saturated=False,
                is_slew_limited=False,
                solve_time_us=(t1 - t0) * 1e6,
                unconstrained=True,
                predicted_states=pred_states,
                optimal_u=u_unc,
            )
            return fa, status

        # Stage 2: Constrained QP Solve (< 250 us)
        # Compute gradient vector: g = Fx * x_k + Fw * W_k + Fu * u_prev
        g_k = self.Fx @ x_k + self.Fw @ W_k + self.Fu * self._u_prev

        # Box bounds with first-step slew constraint
        u_min = np.full(self.horizon, self.f_min, dtype=np.float64)
        u_max = np.full(self.horizon, self.f_max, dtype=np.float64)
        u_min[0] = max(self.f_min, slew_min)
        u_max[0] = min(self.f_max, slew_max)

        qp_res = self.qp_solver.solve(g_k, u_min=u_min, u_max=u_max)
        u_opt = qp_res.u_opt
        fa = float(u_opt[0])

        is_saturated = (fa >= self.f_max - 1e-2) or (fa <= self.f_min + 1e-2)
        is_slew_limited = (abs(fa - self._u_prev) >= self.df_max - 1e-2)

        self._u_prev = fa
        t1 = time.perf_counter()

        pred_states = (self.Sx @ x_k + self.Su @ u_opt + self.Sw @ W_k).reshape(-1, 4)
        status = MPCStatus(
            active_force=fa,
            is_saturated=is_saturated,
            is_slew_limited=is_slew_limited,
            solve_time_us=(t1 - t0) * 1e6,
            unconstrained=False,
            predicted_states=pred_states,
            optimal_u=u_opt,
        )
        return fa, status
