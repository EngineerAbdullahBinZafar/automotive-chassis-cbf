"""
Automated Verification Suite for Performance Benchmarks and Real-Time Latency Budgets.
"""

from __future__ import annotations

import pytest

from src.benchmark.latency_benchmarker import LatencyBenchmarker
from src.benchmark.rms_evaluator import SuspensionBenchmarker
from src.plant.road_profiles import RoadClass


class TestPerformanceBenchmarks:
    """Test suite for Milestone 4 Benchmarking & Latency Verification."""

    def test_iso8608_class_a_benchmark(self) -> None:
        """Verify >= 25% RMS vertical acceleration reduction on ISO Class A road."""
        bench = SuspensionBenchmarker(dt=0.001, ctrl_dt=0.010)
        res = bench.benchmark_iso_road(road_class=RoadClass.CLASS_A, velocity=20.0, duration=4.0)

        pass_m = res["passive"]
        act_m = res["active_mpc_cbf"]

        print(f"\nClass A: Passive RMS={pass_m.rms_sprung_accel:.4f}, Active RMS={act_m.rms_sprung_accel:.4f}, Red={act_m.comfort_improvement_pct:.2f}%")
        assert act_m.comfort_improvement_pct >= 25.0
        assert act_m.max_suspension_stroke <= 0.050
        assert act_m.min_tire_normal_force >= 0.0

    def test_stroke_and_tire_safety_invariants(self) -> None:
        """Verify strict invariance of stroke and tire ground contact under severe excitation."""
        bench = SuspensionBenchmarker(dt=0.001, ctrl_dt=0.010)
        res = bench.benchmark_iso_road(road_class=RoadClass.CLASS_C, velocity=15.0, duration=3.0)

        act_m = res["active_mpc_cbf"]
        assert act_m.max_suspension_stroke <= 0.050
        assert act_m.min_tire_normal_force >= 0.0

    def test_realtime_execution_latency_distribution(self) -> None:
        """Verify real-time execution budget: median latency < 500 us, max < 1000 us."""
        lat_bench = LatencyBenchmarker()
        results = lat_bench.run_benchmark(n_cycles=500)

        total_p = results["total_e2e"]
        print(f"\nTotal E2E: p50={total_p.p50_us:.1f} us, p95={total_p.p95_us:.1f} us, Max={total_p.max_us:.1f} us")
        assert total_p.p50_us < 500.0, f"Median latency {total_p.p50_us:.1f} us exceeds 500 us budget"
        assert total_p.p95_us < 1000.0, f"p95 latency {total_p.p95_us:.1f} us exceeds 1.0 ms budget"
