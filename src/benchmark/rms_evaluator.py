"""
Comprehensive RMS Ride Comfort and Road Holding Evaluator.
Benchmarks Passive Baseline vs Semi-Active Skyhook vs Active MPC+CBF
across ISO 8608 Stochastic Road Profiles (Classes A-E) and Discrete Obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.controller import ActiveSuspensionController, ControllerStepOutput
from src.plant.parameters import QuarterCarParameters, DEFAULT_QUARTER_CAR_PARAMS
from src.plant.quarter_car import QuarterCarModel, QuarterCarState
from src.plant.road_profiles import (
    CombinedRoadProfile,
    ISO8608RoadGenerator,
    Pothole,
    RoadClass,
    SpeedBump,
)


@dataclass(frozen=True)
class SimulationMetrics:
    """Summary metrics for a suspension simulation run."""
    rms_sprung_accel: float         # [m/s^2] RMS body vertical acceleration (ISO 2631-1 comfort)
    peak_sprung_accel: float        # [m/s^2] Peak body acceleration
    rms_suspension_deflection: float# [m] RMS suspension deflection
    max_suspension_stroke: float    # [m] Peak suspension stroke (bound: 0.050 m)
    min_tire_normal_force: float    # [N] Minimum tire ground contact force (bound: >= 0 N)
    comfort_improvement_pct: float  # [%] Reduction in RMS acceleration vs passive baseline
    control_energy: float           # [J] Integral of active force squared: sum(F_a^2 * dt)


class SuspensionBenchmarker:
    """
    Automated benchmark suite evaluating Quarter-Car performance.
    Runs standardized comparative experiments across:
    1. Passive suspension
    2. Semi-active Skyhook suspension
    3. Active MPC + High-Order CBF suspension
    """

    def __init__(
        self,
        params: Optional[QuarterCarParameters] = None,
        dt: float = 0.001,
        ctrl_dt: float = 0.010,
    ) -> None:
        self.params = params if params is not None else DEFAULT_QUARTER_CAR_PARAMS
        self.dt = float(dt)
        self.ctrl_dt = float(ctrl_dt)
        self.ctrl_ratio = max(1, int(round(self.ctrl_dt / self.dt)))

    def run_quarter_car_simulation(
        self,
        mode: str,  # 'passive', 'skyhook', 'active_mpc_cbf'
        elevations: np.ndarray,
        velocities: np.ndarray,
        controller: Optional[ActiveSuspensionController] = None,
        c_sky: float = 1200.0,
    ) -> Tuple[QuarterCarModel, List[QuarterCarState], List[float]]:
        """Run single quarter-car simulation and return model, states, and active forces."""
        p = self.params
        model = QuarterCarModel(p)
        n_steps = len(elevations)
        states: List[QuarterCarState] = []
        forces: List[float] = []

        if mode == "active_mpc_cbf" and controller is None:
            controller = ActiveSuspensionController(params=p, dt=self.ctrl_dt)

        f_cmd = 0.0
        for k in range(n_steps):
            elev = float(elevations[k])
            vel = float(velocities[k])

            if mode == "passive":
                st = model.step_passive(self.dt, elev, vel)
                forces.append(0.0)

            elif mode == "skyhook":
                if k % self.ctrl_ratio == 0:
                    st = model.state
                    # Skyhook damping law: F_sky = -c_sky * dz_s
                    f_cmd = float(np.clip(-c_sky * st.sprung_velocity, p.force_min, p.force_max))
                st = model.step(f_cmd, self.dt, elev, vel)
                forces.append(f_cmd)

            elif mode == "active_mpc_cbf":
                if k % self.ctrl_ratio == 0 and controller is not None:
                    # Provide 12-step road preview lookahead
                    lookahead = [
                        float(velocities[k + j * self.ctrl_ratio])
                        for j in range(12)
                        if k + j * self.ctrl_ratio < n_steps
                    ]
                    out = controller.step(
                        state=model.state,
                        road_preview=lookahead if len(lookahead) == 12 else None,
                        road_elevation=elev,
                        road_velocity=vel,
                    )
                    f_cmd = out.force_command
                st = model.step(f_cmd, self.dt, elev, vel)
                forces.append(f_cmd)

            states.append(st)

        return model, states, forces

    def compute_metrics(
        self,
        states: List[QuarterCarState],
        forces: List[float],
        passive_rms_accel: Optional[float] = None,
        skip_duration: float = 0.5,
    ) -> SimulationMetrics:
        """Calculate standardized engineering metrics from state trajectory."""
        skip = int(skip_duration / self.dt)
        eval_states = states[skip:] if len(states) > skip else states
        eval_forces = forces[skip:] if len(forces) > skip else forces

        accels = np.array([st.sprung_accel for st in eval_states], dtype=np.float64)
        deflections = np.array([st.suspension_deflection for st in eval_states], dtype=np.float64)
        tire_defls = np.array([st.tire_deflection for st in eval_states], dtype=np.float64)
        f_arr = np.array(eval_forces, dtype=np.float64)

        p = self.params
        rms_acc = float(np.sqrt(np.mean(accels**2)))
        peak_acc = float(np.max(np.abs(accels)))
        rms_defl = float(np.sqrt(np.mean(deflections**2)))
        max_stroke = float(np.max(np.abs(deflections)))

        # Tire contact normal force: F_z = F_z_static - k_t * tire_deflection
        tire_forces = p.f_z_static - p.kt * tire_defls
        min_tf = float(np.min(tire_forces))

        if passive_rms_accel is not None and passive_rms_accel > 1e-6:
            reduction = (passive_rms_accel - rms_acc) / passive_rms_accel * 100.0
        else:
            reduction = 0.0

        energy = float(np.sum(f_arr**2) * self.dt)

        return SimulationMetrics(
            rms_sprung_accel=rms_acc,
            peak_sprung_accel=peak_acc,
            rms_suspension_deflection=rms_defl,
            max_suspension_stroke=max_stroke,
            min_tire_normal_force=min_tf,
            comfort_improvement_pct=reduction,
            control_energy=energy,
        )

    def benchmark_iso_road(
        self,
        road_class: RoadClass = RoadClass.CLASS_A,
        velocity: float = 20.0,
        duration: float = 5.0,
        seed: int = 42,
    ) -> Dict[str, SimulationMetrics]:
        """Run full comparative benchmark across Passive, Skyhook, and Active MPC+CBF."""
        gen = ISO8608RoadGenerator(road_class=road_class, velocity=velocity, dt=self.dt, seed=seed)
        _, elevs, vels = gen.generate_profile(duration)

        # 1. Passive baseline
        _, states_pass, forces_pass = self.run_quarter_car_simulation("passive", elevs, vels)
        metrics_pass = self.compute_metrics(states_pass, forces_pass)

        # 2. Semi-active Skyhook
        _, states_sky, forces_sky = self.run_quarter_car_simulation("skyhook", elevs, vels)
        metrics_sky = self.compute_metrics(states_sky, forces_sky, passive_rms_accel=metrics_pass.rms_sprung_accel)

        # 3. Active MPC + CBF
        ctrl = ActiveSuspensionController(params=self.params, dt=self.ctrl_dt)
        _, states_act, forces_act = self.run_quarter_car_simulation("active_mpc_cbf", elevs, vels, controller=ctrl)
        metrics_act = self.compute_metrics(states_act, forces_act, passive_rms_accel=metrics_pass.rms_sprung_accel)

        return {
            "passive": metrics_pass,
            "skyhook": metrics_sky,
            "active_mpc_cbf": metrics_act,
        }
