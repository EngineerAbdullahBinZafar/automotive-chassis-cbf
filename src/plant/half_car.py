"""
Half-Car 4-DOF Pitch-Bounce Active Suspension Plant Model.
Executive Luxury Sedan Chassis Scale (BMW 7-Series / Rolls-Royce Ghost).

State vector (8 states):
    x[0] = zs         : Chassis CG vertical displacement [m] (upward positive)
    x[1] = zs_dot     : Chassis CG vertical velocity [m/s]
    x[2] = theta      : Chassis pitch angle [rad] (nose-down positive)
    x[3] = theta_dot  : Chassis pitch angular velocity [rad/s]
    x[4] = zuf        : Front unsprung mass vertical displacement [m]
    x[5] = zuf_dot    : Front unsprung mass vertical velocity [m/s]
    x[6] = zur        : Rear unsprung mass vertical displacement [m]
    x[7] = zur_dot    : Rear unsprung mass vertical velocity [m/s]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from src.plant.parameters import HalfCarParameters, DEFAULT_HALF_CAR_PARAMS


@dataclass
class HalfCarState:
    """
    Contractual Half-Car state output representation.
    Adheres strictly to PROJECT.md line 72 interface specification:
    HalfCarState(zs, zs_dot, theta, theta_dot, zuf, zuf_dot, zur, zur_dot, zs_accel, theta_accel)
    """
    zs: float                  # Chassis CG displacement [m]
    zs_dot: float              # Chassis CG velocity [m/s]
    theta: float               # Chassis pitch angle [rad] (nose-down > 0)
    theta_dot: float           # Chassis pitch rate [rad/s]
    zuf: float                 # Front unsprung displacement [m]
    zuf_dot: float             # Front unsprung velocity [m/s]
    zur: float                 # Rear unsprung displacement [m]
    zur_dot: float             # Rear unsprung velocity [m/s]
    zs_accel: float = 0.0      # Chassis CG vertical acceleration [m/s^2]
    theta_accel: float = 0.0   # Chassis pitch angular acceleration [rad/s^2]
    zuf_accel: float = 0.0     # Front wheel vertical acceleration [m/s^2]
    zur_accel: float = 0.0     # Rear wheel vertical acceleration [m/s^2]

    # Kinematics and deflections
    z_sf: float = 0.0          # Front attachment: zs - a*theta [m]
    z_sr: float = 0.0          # Rear attachment: zs + b*theta [m]
    susp_defl_front: float = 0.0 # Front stroke: z_sf - zuf [m]
    susp_defl_rear: float = 0.0  # Rear stroke: z_sr - zur [m]
    tire_defl_front: float = 0.0 # Front tire deflection: zuf - zrf [m]
    tire_defl_rear: float = 0.0  # Rear tire deflection: zur - zrr [m]

    # Forces
    f_sf: float = 0.0          # Front net suspension upward force on body [N]
    f_sr: float = 0.0          # Rear net suspension upward force on body [N]
    f_zf_total: float = 0.0    # Front total tire ground contact force [N]
    f_zr_total: float = 0.0    # Rear total tire ground contact force [N]

    # Constraint flags
    stroke_violation_front: bool = False
    stroke_violation_rear: bool = False
    tire_liftoff_front: bool = False
    tire_liftoff_rear: bool = False

    def to_list(self) -> List[float]:
        """Return 8-element state vector as list."""
        return [
            self.zs, self.zs_dot, self.theta, self.theta_dot,
            self.zuf, self.zuf_dot, self.zur, self.zur_dot,
        ]

    def to_tuple(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return 8-element state vector as tuple."""
        return (
            self.zs, self.zs_dot, self.theta, self.theta_dot,
            self.zuf, self.zuf_dot, self.zur, self.zur_dot,
        )

    def to_numpy(self) -> Any:
        """Return 8-element state vector as NumPy array."""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy not available")
        return np.array(self.to_list(), dtype=np.float64)


class HalfCarModel:
    """
    4-DOF Deterministic Half-Car Pitch-Bounce Plant Model.
    Executive luxury sedan scale with Maurice Olley pitch-bounce isolation (fp/fb = 1.32).
    """

    def __init__(
        self,
        params: Optional[HalfCarParameters] = None,
        initial_state: Optional[HalfCarState] = None,
    ) -> None:
        self.params = params if params is not None else DEFAULT_HALF_CAR_PARAMS
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = HalfCarState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._time = 0.0
        self._last_u = (0.0, 0.0)
        self._last_w = (0.0, 0.0)
        self._init_state_space()

    def _init_state_space(self) -> None:
        """Pre-compute continuous state-space matrices A (8x8), B (8x2), G (8x2)."""
        p = self.params
        A: List[List[float]] = [[0.0] * 8 for _ in range(8)]
        A[0][1] = 1.0
        A[1][0] = -(p.ksf + p.ksr) / p.ms
        A[1][1] = -(p.csf + p.csr) / p.ms
        A[1][2] = (p.a * p.ksf - p.b * p.ksr) / p.ms
        A[1][3] = (p.a * p.csf - p.b * p.csr) / p.ms
        A[1][4] = p.ksf / p.ms
        A[1][5] = p.csf / p.ms
        A[1][6] = p.ksr / p.ms
        A[1][7] = p.csr / p.ms

        A[2][3] = 1.0
        A[3][0] = (p.a * p.ksf - p.b * p.ksr) / p.iy
        A[3][1] = (p.a * p.csf - p.b * p.csr) / p.iy
        A[3][2] = -(p.a**2 * p.ksf + p.b**2 * p.ksr) / p.iy
        A[3][3] = -(p.a**2 * p.csf + p.b**2 * p.csr) / p.iy
        A[3][4] = -p.a * p.ksf / p.iy
        A[3][5] = -p.a * p.csf / p.iy
        A[3][6] = p.b * p.ksr / p.iy
        A[3][7] = p.b * p.csr / p.iy

        A[4][5] = 1.0
        A[5][0] = p.ksf / p.muf
        A[5][1] = p.csf / p.muf
        A[5][2] = -p.a * p.ksf / p.muf
        A[5][3] = -p.a * p.csf / p.muf
        A[5][4] = -(p.ksf + p.ktf) / p.muf
        A[5][5] = -p.csf / p.muf

        A[6][7] = 1.0
        A[7][0] = p.ksr / p.mur
        A[7][1] = p.csr / p.mur
        A[7][2] = p.b * p.ksr / p.mur
        A[7][3] = p.b * p.csr / p.mur
        A[7][6] = -(p.ksr + p.ktr) / p.mur
        A[7][7] = -p.csr / p.mur

        B: List[List[float]] = [[0.0, 0.0] for _ in range(8)]
        B[1][0] = 1.0 / p.ms
        B[1][1] = 1.0 / p.ms
        B[3][0] = -p.a / p.iy
        B[3][1] = p.b / p.iy
        B[5][0] = -1.0 / p.muf
        B[7][1] = -1.0 / p.mur

        G: List[List[float]] = [[0.0, 0.0] for _ in range(8)]
        G[5][0] = p.ktf / p.muf
        G[7][1] = p.ktr / p.mur

        self._A_list = A
        self._B_list = B
        self._G_list = G

        if HAS_NUMPY:
            self.A = np.array(A, dtype=np.float64)
            self.B = np.array(B, dtype=np.float64)
            self.G = np.array(G, dtype=np.float64)
        else:
            self.A = A
            self.B = B
            self.G = G

    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return self._time

    def get_continuous_state_space(self) -> Dict[str, Any]:
        """Return state-space matrices A, B, G in dictionary."""
        return {
            "A": self._A_list,
            "B": self._B_list,
            "G": self._G_list,
        }

    def derivatives(
        self,
        x: Sequence[float],
        u: Tuple[float, float],
        w: Tuple[float, float],
    ) -> List[float]:
        """
        Pure-Python scalar Newton-Euler derivative evaluation (O(1), ~18 us).

        Args:
            x: State sequence [zs, dzs, theta, dtheta, zuf, dzuf, zur, dzur]
            u: Actuator forces (F_af, F_ar) [N]
            w: Road elevations (z_rf, z_rr) [m]
        """
        p = self.params
        zs, zs_dot, theta, theta_dot, zuf, zuf_dot, zur, zur_dot = x
        F_af, F_ar = u
        z_rf, z_rr = w

        # Body attachment kinematics (small-angle approximation)
        z_sf = zs - p.a * theta
        z_sf_dot = zs_dot - p.a * theta_dot
        z_sr = zs + p.b * theta
        z_sr_dot = zs_dot + p.b * theta_dot

        # Net upward suspension forces exerted on sprung mass
        F_sf = p.ksf * (zuf - z_sf) + p.csf * (zuf_dot - z_sf_dot) + F_af
        F_sr = p.ksr * (zur - z_sr) + p.csr * (zur_dot - z_sr_dot) + F_ar

        # Accelerations
        zs_ddot = (F_sf + F_sr) / p.ms
        theta_ddot = (-p.a * F_sf + p.b * F_sr) / p.iy
        zuf_ddot = (-F_sf - p.ktf * (zuf - z_rf)) / p.muf
        zur_ddot = (-F_sr - p.ktr * (zur - z_rr)) / p.mur

        return [zs_dot, zs_ddot, theta_dot, theta_ddot, zuf_dot, zuf_ddot, zur_dot, zur_ddot]

    def step(
        self,
        active_forces: Tuple[float, float],
        dt: float,
        road_elevations: Tuple[float, float],
        road_velocities: Optional[Tuple[float, float]] = None,
    ) -> HalfCarState:
        """
        Execute deterministic fixed-step 4th-Order Runge-Kutta (RK4) integration step.
        """
        p = self.params
        f_min, f_max = p.force_min, p.force_max
        F_af = max(f_min, min(f_max, float(active_forces[0])))
        F_ar = max(f_min, min(f_max, float(active_forces[1])))
        u = (F_af, F_ar)
        w = (float(road_elevations[0]), float(road_elevations[1]))

        x0 = self.state.to_list()

        # RK4 stages
        k1 = self.derivatives(x0, u, w)
        x_k2 = [x0[i] + 0.5 * dt * k1[i] for i in range(8)]
        k2 = self.derivatives(x_k2, u, w)
        x_k3 = [x0[i] + 0.5 * dt * k2[i] for i in range(8)]
        k3 = self.derivatives(x_k3, u, w)
        x_k4 = [x0[i] + dt * k3[i] for i in range(8)]
        k4 = self.derivatives(x_k4, u, w)

        x_next = [
            x0[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            for i in range(8)
        ]

        # Evaluate derivatives and forces at new state
        derivs = self.derivatives(x_next, u, w)
        zs_ddot = derivs[1]
        theta_ddot = derivs[3]
        zuf_ddot = derivs[5]
        zur_ddot = derivs[7]

        zs, zs_dot, theta, theta_dot, zuf, zuf_dot, zur, zur_dot = x_next
        z_sf = zs - p.a * theta
        z_sf_dot = zs_dot - p.a * theta_dot
        z_sr = zs + p.b * theta
        z_sr_dot = zs_dot + p.b * theta_dot

        d_sf = z_sf - zuf
        d_sr = z_sr - zur
        d_tf = zuf - w[0]
        d_tr = zur - w[1]

        F_sf = p.ksf * (zuf - z_sf) + p.csf * (zuf_dot - z_sf_dot) + F_af
        F_sr = p.ksr * (zur - z_sr) + p.csr * (zur_dot - z_sr_dot) + F_ar

        raw_f_zf = p.f_zf_static - p.ktf * d_tf
        raw_f_zr = p.f_zr_static - p.ktr * d_tr
        F_zf_total = max(0.0, raw_f_zf)
        F_zr_total = max(0.0, raw_f_zr)

        stroke_lim = p.stroke_max
        self.state = HalfCarState(
            zs=zs,
            zs_dot=zs_dot,
            theta=theta,
            theta_dot=theta_dot,
            zuf=zuf,
            zuf_dot=zuf_dot,
            zur=zur,
            zur_dot=zur_dot,
            zs_accel=zs_ddot,
            theta_accel=theta_ddot,
            zuf_accel=zuf_ddot,
            zur_accel=zur_ddot,
            z_sf=z_sf,
            z_sr=z_sr,
            susp_defl_front=d_sf,
            susp_defl_rear=d_sr,
            tire_defl_front=d_tf,
            tire_defl_rear=d_tr,
            f_sf=F_sf,
            f_sr=F_sr,
            f_zf_total=F_zf_total,
            f_zr_total=F_zr_total,
            stroke_violation_front=abs(d_sf) > stroke_lim,
            stroke_violation_rear=abs(d_sr) > stroke_lim,
            tire_liftoff_front=raw_f_zf <= 0.0,
            tire_liftoff_rear=raw_f_zr <= 0.0,
        )

        self._time += dt
        self._last_u = u
        self._last_w = w

        return self.state

    def step_passive(
        self,
        dt: float,
        road_elevations: Tuple[float, float],
        road_velocities: Optional[Tuple[float, float]] = None,
    ) -> HalfCarState:
        """Execute step simulation with actuator passive baseline (F_af = 0, F_ar = 0)."""
        return self.step(
            active_forces=(0.0, 0.0),
            dt=dt,
            road_elevations=road_elevations,
            road_velocities=road_velocities,
        )

    def reset(self, initial_state: Optional[HalfCarState] = None) -> HalfCarState:
        """Reset state vector to zeros or user-provided state."""
        self.state = initial_state or HalfCarState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._time = 0.0
        self._last_u = (0.0, 0.0)
        self._last_w = (0.0, 0.0)
        return self.state

    def check_constraints(
        self,
        state: Optional[HalfCarState] = None,
        active_forces: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate physical constraints: stroke clearance, tire contact, and actuator limits.
        """
        st = state if state is not None else self.state
        u = active_forces if active_forces is not None else self._last_u
        p = self.params

        d_sf = st.susp_defl_front
        d_sr = st.susp_defl_rear
        stroke_lim = p.stroke_max

        stroke_viol_f = abs(d_sf) > stroke_lim
        stroke_viol_r = abs(d_sr) > stroke_lim

        margin_f = min(stroke_lim - d_sf, d_sf - (-stroke_lim))
        margin_r = min(stroke_lim - d_sr, d_sr - (-stroke_lim))

        raw_f_zf = p.f_zf_static - p.ktf * st.tire_defl_front
        raw_f_zr = p.f_zr_static - p.ktr * st.tire_defl_rear

        liftoff_f = raw_f_zf <= 0.0
        liftoff_r = raw_f_zr <= 0.0

        sat_f = (u[0] >= p.force_max) or (u[0] <= p.force_min)
        sat_r = (u[1] >= p.force_max) or (u[1] <= p.force_min)

        return {
            "stroke_violation_front": stroke_viol_f,
            "stroke_violation_rear": stroke_viol_r,
            "stroke_margin_front": margin_f,
            "stroke_margin_rear": margin_r,
            "tire_liftoff_front": liftoff_f,
            "tire_liftoff_rear": liftoff_r,
            "tire_force_front": max(0.0, raw_f_zf),
            "tire_force_rear": max(0.0, raw_f_zr),
            "actuator_saturated_front": sat_f,
            "actuator_saturated_rear": sat_r,
        }
