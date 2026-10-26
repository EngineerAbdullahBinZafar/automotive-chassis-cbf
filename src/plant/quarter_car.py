"""
2-DOF Quarter-Car Active Suspension Dynamics Plant Model.
Fully supports pure Python standard library and vectorized NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from src.plant.parameters import QuarterCarParameters, DEFAULT_QUARTER_CAR_PARAMS


@dataclass(frozen=True)
class QuarterCarState:
    """
    Quarter-Car state output representation.
    Adheres strictly to PROJECT.md line 71 interface specification:
    QuarterCarState(suspension_deflection, sprung_velocity, tire_deflection, unsprung_velocity, sprung_accel)
    """
    suspension_deflection: float  # z_s - z_u [m] (extension/rebound > 0, compression/jounce < 0)
    sprung_velocity: float        # z_dot_s [m/s]
    tire_deflection: float        # z_u - z_r [m] (tire extension > 0, compression < 0)
    unsprung_velocity: float      # z_dot_u [m/s]
    sprung_accel: float           # z_ddot_s [m/s^2]

    def to_tuple(self) -> Tuple[float, float, float, float, float]:
        """Convert state to 5-element tuple."""
        return (
            self.suspension_deflection,
            self.sprung_velocity,
            self.tire_deflection,
            self.unsprung_velocity,
            self.sprung_accel,
        )

    def to_relative_vector(self) -> Tuple[float, float, float, float]:
        """Convert state to 4-element control state vector x_rel."""
        return (
            self.suspension_deflection,
            self.sprung_velocity,
            self.tire_deflection,
            self.unsprung_velocity,
        )

    def to_numpy(self) -> Any:
        """Convert state to NumPy array."""
        if not HAS_NUMPY:
            raise RuntimeError("NumPy not available")
        return np.array(self.to_tuple(), dtype=np.float64)


@dataclass(frozen=True)
class ConstraintStatus:
    """Physical constraint evaluation status."""
    stroke_violation: bool        # True if |z_s - z_u| > stroke_max
    tire_liftoff: bool            # True if F_z_total <= 0.0 (ground contact lost)
    stroke_margin: float          # Remaining stroke distance to boundary [m]
    tire_force_total: float       # Ground contact force max(0, F_z_static - kt*(z_u - z_r)) [N]
    tire_load_ratio: float        # Dynamic load variation / static preload (>= 1.0 indicates liftoff)
    actuator_saturated: bool      # True if |F_a| >= F_a_max


class QuarterCarModel:
    """
    2-DOF Deterministic Quarter-Car Vehicle Dynamics Model.

    States:
      Absolute:  x_abs = [z_s, z_dot_s, z_u, z_dot_u]
      Relative:  x_rel = [z_s - z_u, z_dot_s, z_u - z_r, z_dot_u]
    """

    def __init__(
        self,
        params: Optional[QuarterCarParameters] = None,
        initial_zs: float = 0.0,
        initial_dzs: float = 0.0,
        initial_zu: float = 0.0,
        initial_dzu: float = 0.0,
    ) -> None:
        self.params = params if params is not None else DEFAULT_QUARTER_CAR_PARAMS
        self._zs = float(initial_zs)
        self._dzs = float(initial_dzs)
        self._zu = float(initial_zu)
        self._dzu = float(initial_dzu)
        self._time = 0.0
        self._last_zr = 0.0
        self._last_dzr = 0.0
        self._last_fa = 0.0

    def reset(
        self,
        zs: float = 0.0,
        dzs: float = 0.0,
        zu: float = 0.0,
        dzu: float = 0.0,
    ) -> None:
        """Reset state variables to prescribed initial conditions."""
        self._zs = float(zs)
        self._dzs = float(dzs)
        self._zu = float(zu)
        self._dzu = float(dzu)
        self._time = 0.0
        self._last_zr = 0.0
        self._last_dzr = 0.0
        self._last_fa = 0.0

    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return self._time

    @property
    def absolute_state(self) -> Tuple[float, float, float, float]:
        """Absolute state tuple (z_s, z_dot_s, z_u, z_dot_u)."""
        return (self._zs, self._dzs, self._zu, self._dzu)

    @property
    def state(self) -> QuarterCarState:
        """Current QuarterCarState output."""
        susp_defl = self._zs - self._zu
        tire_defl = self._zu - self._last_zr
        p = self.params
        sprung_acc = (
            -p.ks * susp_defl - p.cs * (self._dzs - self._dzu) + self._last_fa
        ) / p.ms
        return QuarterCarState(
            suspension_deflection=susp_defl,
            sprung_velocity=self._dzs,
            tire_deflection=tire_defl,
            unsprung_velocity=self._dzu,
            sprung_accel=sprung_acc,
        )

    def derivatives_absolute(
        self,
        state: Tuple[float, float, float, float],
        active_force: float,
        road_elevation: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute continuous-time derivatives for absolute state:
          dx_abs/dt = A_abs * x_abs + B_abs * active_force + G_abs * road_elevation
        """
        zs, dzs, zu, dzu = state
        p = self.params
        ddzs = (-p.ks * (zs - zu) - p.cs * (dzs - dzu) + active_force) / p.ms
        ddzu = (
            p.ks * (zs - zu)
            + p.cs * (dzs - dzu)
            - p.kt * (zu - road_elevation)
            - p.ct * dzu
            - active_force
        ) / p.mu
        return (dzs, ddzs, dzu, ddzu)

    def derivatives_relative(
        self,
        state: Tuple[float, float, float, float],
        active_force: float,
        road_velocity: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute continuous-time derivatives for relative state:
          dx_rel/dt = A_rel * x_rel + B_rel * active_force + G_rel * road_velocity
        """
        x1, x2, x3, x4 = state  # (z_s - z_u, z_dot_s, z_u - z_r, z_dot_u)
        p = self.params
        dx1 = x2 - x4
        dx2 = (-p.ks * x1 - p.cs * (x2 - x4) + active_force) / p.ms
        dx3 = x4 - road_velocity
        dx4 = (p.ks * x1 + p.cs * (x2 - x4) - p.kt * x3 - active_force) / p.mu
        return (dx1, dx2, dx3, dx4)

    def step(
        self,
        active_force: float,
        dt: float,
        road_elevation: float,
        road_velocity: float = 0.0,
    ) -> QuarterCarState:
        """
        Deterministic fixed-step 4th-Order Runge-Kutta (RK4) integration step.
        Updates internal state by dt and returns the new QuarterCarState.
        """
        s0 = (self._zs, self._dzs, self._zu, self._dzu)

        # Stage 1
        k1 = self.derivatives_absolute(s0, active_force, road_elevation)

        # Stage 2
        s1 = (
            s0[0] + 0.5 * dt * k1[0],
            s0[1] + 0.5 * dt * k1[1],
            s0[2] + 0.5 * dt * k1[2],
            s0[3] + 0.5 * dt * k1[3],
        )
        k2 = self.derivatives_absolute(s1, active_force, road_elevation)

        # Stage 3
        s2 = (
            s0[0] + 0.5 * dt * k2[0],
            s0[1] + 0.5 * dt * k2[1],
            s0[2] + 0.5 * dt * k2[2],
            s0[3] + 0.5 * dt * k2[3],
        )
        k3 = self.derivatives_absolute(s2, active_force, road_elevation)

        # Stage 4
        s3 = (
            s0[0] + dt * k3[0],
            s0[1] + dt * k3[1],
            s0[2] + dt * k3[2],
            s0[3] + dt * k3[3],
        )
        k4 = self.derivatives_absolute(s3, active_force, road_elevation)

        # Update internal states
        self._zs += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        self._dzs += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        self._zu += (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        self._dzu += (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])

        self._time += dt
        self._last_zr = road_elevation
        self._last_dzr = road_velocity
        self._last_fa = active_force

        return self.state

    def step_passive(
        self,
        dt: float,
        road_elevation: float,
        road_velocity: float = 0.0,
    ) -> QuarterCarState:
        """Execute step simulation with actuator passive baseline (F_a = 0.0)."""
        return self.step(
            active_force=0.0,
            dt=dt,
            road_elevation=road_elevation,
            road_velocity=road_velocity,
        )

    def check_constraints(
        self,
        state: Optional[QuarterCarState] = None,
        active_force: Optional[float] = None,
    ) -> ConstraintStatus:
        """
        Evaluate physical constraints: stroke limits, tire liftoff, and actuator saturation.
        """
        st = state if state is not None else self.state
        fa = active_force if active_force is not None else self._last_fa
        p = self.params

        susp_defl = st.suspension_deflection
        tire_defl = st.tire_deflection

        # 1. Stroke check
        stroke_violation = (susp_defl > p.stroke_max) or (susp_defl < p.stroke_min)
        stroke_margin = min(p.stroke_max - susp_defl, susp_defl - p.stroke_min)

        # 2. Tire liftoff check
        dynamic_tire_force = p.kt * tire_defl
        raw_contact_force = p.f_z_static - dynamic_tire_force
        tire_force_total = max(0.0, raw_contact_force)
        tire_load_ratio = dynamic_tire_force / p.f_z_static
        tire_liftoff = (raw_contact_force <= 0.0)

        # 3. Actuator saturation check
        actuator_saturated = (fa >= p.force_max) or (fa <= p.force_min)

        return ConstraintStatus(
            stroke_violation=stroke_violation,
            tire_liftoff=tire_liftoff,
            stroke_margin=stroke_margin,
            tire_force_total=tire_force_total,
            tire_load_ratio=tire_load_ratio,
            actuator_saturated=actuator_saturated,
        )

    def get_continuous_state_space_relative(self) -> Dict[str, Any]:
        """
        Return relative state-space matrices A, B, G, C, D in nested list format.
        """
        p = self.params
        A = [
            [0.0, 1.0, 0.0, -1.0],
            [-p.ks / p.ms, -p.cs / p.ms, 0.0, p.cs / p.ms],
            [0.0, 0.0, 0.0, 1.0],
            [p.ks / p.mu, p.cs / p.mu, -p.kt / p.mu, -p.cs / p.mu],
        ]
        B = [[0.0], [1.0 / p.ms], [0.0], [-1.0 / p.mu]]
        G = [[0.0], [0.0], [-1.0], [0.0]]
        C = [
            [-p.ks / p.ms, -p.cs / p.ms, 0.0, p.cs / p.ms],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        D = [[1.0 / p.ms], [0.0], [0.0]]
        return {"A": A, "B": B, "G": G, "C": C, "D": D}

    def get_continuous_state_space_absolute(self) -> Dict[str, Any]:
        """
        Return absolute state-space matrices A_abs, B_abs, G_abs.
        """
        p = self.params
        A = [
            [0.0, 1.0, 0.0, 0.0],
            [-p.ks / p.ms, -p.cs / p.ms, p.ks / p.ms, p.cs / p.ms],
            [0.0, 0.0, 0.0, 1.0],
            [p.ks / p.mu, p.cs / p.mu, -(p.ks + p.kt) / p.mu, -p.cs / p.mu],
        ]
        B = [[0.0], [1.0 / p.ms], [0.0], [-1.0 / p.mu]]
        G = [[0.0], [0.0], [0.0], [p.kt / p.mu]]
        return {"A": A, "B": B, "G": G}
