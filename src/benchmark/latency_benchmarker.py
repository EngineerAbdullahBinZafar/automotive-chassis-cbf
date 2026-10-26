"""
High-Resolution Real-Time Latency Benchmarker.
Profiles execution time of Level 1 (MPC), Level 2 (CBF), Level 3 (Actuator),
and Level 4 (CAN DBC serialization) across 1,000+ continuous control cycles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.controller import ActiveSuspensionController
from src.plant.parameters import DEFAULT_QUARTER_CAR_PARAMS, QuarterCarParameters
from src.plant.quarter_car import QuarterCarState
from src.telemetry.dbc_codec import ChassisDbcCodec, ChassisHeaveFrame


@dataclass(frozen=True)
class LatencyPercentiles:
    """Latency distribution summary in microseconds [us]."""
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    mean_us: float
    std_us: float


class LatencyBenchmarker:
    """Automated profiler measuring execution timing distributions."""

    def __init__(self, params: Optional[QuarterCarParameters] = None) -> None:
        self.params = params if params is not None else DEFAULT_QUARTER_CAR_PARAMS
        self.controller = ActiveSuspensionController(params=self.params, dt=0.010)
        self.codec = ChassisDbcCodec()

    def run_benchmark(self, n_cycles: int = 1000) -> Dict[str, LatencyPercentiles]:
        """
        Execute n_cycles of end-to-end control steps and profile breakdown.
        """
        st_test = QuarterCarState(
            suspension_deflection=0.012,
            sprung_velocity=0.15,
            tire_deflection=-0.004,
            unsprung_velocity=-0.08,
            sprung_accel=0.95,
        )

        mpc_times: List[float] = []
        cbf_times: List[float] = []
        can_times: List[float] = []
        total_times: List[float] = []

        # Warm-up JIT and memory cache
        for _ in range(50):
            self.controller.step(st_test)

        for _ in range(n_cycles):
            t_start = time.perf_counter()

            # Level 1: MPC solve
            t0 = time.perf_counter()
            f_mpc, _ = self.controller.mpc.compute_control_with_status(st_test)
            t1 = time.perf_counter()
            mpc_times.append((t1 - t0) * 1e6)

            # Level 2: CBF filter
            t2 = time.perf_counter()
            f_safe, _ = self.controller.cbf.filter_control(f_mpc, st_test)
            t3 = time.perf_counter()
            cbf_times.append((t3 - t2) * 1e6)

            # Level 4: CAN DBC encode
            t4 = time.perf_counter()
            frame = ChassisHeaveFrame(
                sprung_accel=st_test.sprung_accel,
                sprung_velocity=st_test.sprung_velocity,
                suspension_deflection=st_test.suspension_deflection,
                chassis_mode=1,
            )
            _ = self.codec.encode_chassis_heave(frame, protect=True)
            t5 = time.perf_counter()
            can_times.append((t5 - t4) * 1e6)

            t_end = time.perf_counter()
            total_times.append((t_end - t_start) * 1e6)

        return {
            "mpc": self._compute_percentiles(mpc_times),
            "cbf": self._compute_percentiles(cbf_times),
            "can": self._compute_percentiles(can_times),
            "total_e2e": self._compute_percentiles(total_times),
        }

    def _compute_percentiles(self, latencies_us: List[float]) -> LatencyPercentiles:
        """Compute statistical percentiles."""
        arr = np.array(latencies_us, dtype=np.float64)
        return LatencyPercentiles(
            p50_us=float(np.percentile(arr, 50)),
            p95_us=float(np.percentile(arr, 95)),
            p99_us=float(np.percentile(arr, 99)),
            max_us=float(np.max(arr)),
            mean_us=float(np.mean(arr)),
            std_us=float(np.std(arr)),
        )
