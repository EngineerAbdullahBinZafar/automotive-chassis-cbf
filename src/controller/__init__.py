"""
Active Suspension Controller Module (Milestone 2).

Provides:
- Level 1: RideComfortMPC (100 Hz condensed linear MPC planner)
- Level 2: CBFFilter / HighOrderCBFFilter (HOCBF relative degree r=2 safety shield)
- Level 3: ActuatorModel (physical saturation, slew limiter, thermal dynamics, status bitfield)
- Real-Time Solvers: HildrethQPSolver, OSQPDirectSolver, RealTimeQPSolver
- Combined Pipeline: ActiveSuspensionController
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.controller.actuator import (
    ActuatorHealthCode,
    ActuatorMode,
    ActuatorModel,
    ActuatorState,
    ActuatorStatusFlags,
)
from src.controller.cbf_filter import (
    CBFFilter,
    CBFStatus,
    HighOrderCBFFilter,
)
from src.controller.mpc_controller import (
    MPCStatus,
    MPCWeights,
    RideComfortMPC,
)
from src.controller.qp_solvers import (
    HildrethQPSolver,
    OSQPDirectSolver,
    QPSolverResult,
    QPSolverStatus,
    RealTimeQPSolver,
)
from src.plant.parameters import QuarterCarParameters, DEFAULT_QUARTER_CAR_PARAMS
from src.plant.quarter_car import QuarterCarState


@dataclass(frozen=True)
class ControllerStepOutput:
    """End-to-end active suspension controller step output."""
    force_command: float          # Final physical actuator force delivered [N]
    mpc_force: float              # Nominal force planned by MPC [N]
    cbf_force: float              # Safe force projected by CBF filter [N]
    mpc_status: MPCStatus         # Level 1 MPC diagnostics
    cbf_status: CBFStatus         # Level 2 CBF diagnostics
    actuator_state: ActuatorState # Level 3 Actuator physical state
    total_latency_us: float       # Total end-to-end execution latency [us]


class ActiveSuspensionController:
    """
    Unified Hierarchical Active Suspension Controller:
      Level 1: Ride Comfort MPC Planner (minimizes body acceleration)
      Level 2: High-Order CBF Safety Shield (enforces stroke & tire contact)
      Level 3: Physical Actuator Strut (saturation, slew limiting, thermal tracking)
    """

    def __init__(
        self,
        params: Optional[QuarterCarParameters] = None,
        mpc_weights: Optional[MPCWeights] = None,
        mpc_horizon: int = 12,
        dt: float = 0.010,
        enable_cbf: bool = True,
        enable_actuator_dynamics: bool = True,
        solver_type: str = "auto",
    ) -> None:
        self.params = params if params is not None else DEFAULT_QUARTER_CAR_PARAMS
        self.dt = float(dt)
        self.enable_cbf = bool(enable_cbf)
        self.enable_actuator_dynamics = bool(enable_actuator_dynamics)

        self.mpc = RideComfortMPC(
            params=self.params,
            weights=mpc_weights,
            horizon=mpc_horizon,
            ts=self.dt,
            solver_type=solver_type,
        )
        self.cbf = CBFFilter(
            params=self.params,
            dt=self.dt,
        )
        self.actuator = ActuatorModel(
            force_min=self.params.force_min,
            force_max=self.params.force_max,
            slew_rate_max=self.params.slew_rate_max,
        )

    def reset(self) -> None:
        """Reset internal states across all controller levels."""
        self.mpc.reset()
        self.cbf.reset()
        self.actuator.reset()

    def step(
        self,
        state: QuarterCarState,
        road_preview: Optional[Sequence[float]] = None,
        road_elevation: float = 0.0,
        road_velocity: float = 0.0,
        road_accel: float = 0.0,
    ) -> ControllerStepOutput:
        """
        Execute one complete hierarchical control cycle (10 ms).
        Total latency benchmarked < 0.5 ms.
        """
        t0 = time.perf_counter()

        # Level 1: Ride Comfort MPC Planner
        f_mpc, mpc_status = self.mpc.compute_control_with_status(state, road_preview)

        # Level 2: High-Order Control Barrier Function Safety Shield
        if self.enable_cbf:
            f_cbf, cbf_status = self.cbf.filter_control(
                f_ref=f_mpc,
                state=state,
                road_elevation=road_elevation,
                road_velocity=road_velocity,
                road_accel=road_accel,
            )
        else:
            f_cbf = f_mpc
            cbf_status = CBFStatus(
                f_safe=f_mpc,
                f_ref=f_mpc,
                is_safe=True,
                f_min_cbf=self.params.force_min,
                f_max_cbf=self.params.force_max,
                f_min_tire=self.params.force_min,
                slew_lower=self.params.force_min,
                slew_upper=self.params.force_max,
                lower_bound=self.params.force_min,
                upper_bound=self.params.force_max,
                slack=0.0,
                active_constraints=(),
                execution_time_us=0.0,
            )

        # Level 3: Physical Actuator Strut Dynamics
        susp_vel = float(state.sprung_velocity - state.unsprung_velocity)
        if self.enable_actuator_dynamics:
            actuator_state = self.actuator.step(f_cbf, dt=self.dt, susp_vel=susp_vel)
            f_final = actuator_state.force
        else:
            f_final = f_cbf
            actuator_state = self.actuator.step(f_cbf, dt=self.dt, susp_vel=susp_vel)

        t1 = time.perf_counter()
        total_latency_us = (t1 - t0) * 1e6

        return ControllerStepOutput(
            force_command=f_final,
            mpc_force=f_mpc,
            cbf_force=f_cbf,
            mpc_status=mpc_status,
            cbf_status=cbf_status,
            actuator_state=actuator_state,
            total_latency_us=total_latency_us,
        )


__all__ = [
    "RideComfortMPC",
    "MPCWeights",
    "MPCStatus",
    "CBFFilter",
    "HighOrderCBFFilter",
    "CBFStatus",
    "HildrethQPSolver",
    "OSQPDirectSolver",
    "RealTimeQPSolver",
    "QPSolverStatus",
    "QPSolverResult",
    "ActuatorModel",
    "ActuatorState",
    "ActuatorStatusFlags",
    "ActuatorMode",
    "ActuatorHealthCode",
    "ActiveSuspensionController",
    "ControllerStepOutput",
]
