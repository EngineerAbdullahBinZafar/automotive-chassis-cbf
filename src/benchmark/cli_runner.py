"""
Interactive Automotive Chassis CLI Runner and Performance Demonstrator.
Run real-time vehicle dynamics simulations, ISO 8608 roughness benchmarks,
and inspect active suspension performance in the terminal.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import numpy as np

from src.benchmark.latency_benchmarker import LatencyBenchmarker
from src.benchmark.rms_evaluator import SuspensionBenchmarker
from src.plant.parameters import DEFAULT_QUARTER_CAR_PARAMS
from src.plant.road_profiles import Pothole, RoadClass, SpeedBump


def print_banner() -> None:
    """Print industrial terminal banner."""
    print("=" * 78)
    print("  BMW / ROLLS-ROYCE RESEARCH CHASSIS CONTROL & HOCBF SAFETY SHIELD")
    print("  Real-Time 100 Hz Model Predictive Control & Control Barrier Functions")
    print("=" * 78)
    print("  Vehicle Class   : Ultra-Luxury Executive Sedan (Ghost / 7-Series)")
    print(f"  Sprung Mass     : {DEFAULT_QUARTER_CAR_PARAMS.ms:.1f} kg (Corner)")
    print(f"  Unsprung Mass   : {DEFAULT_QUARTER_CAR_PARAMS.mu:.1f} kg")
    print(f"  Actuator Limits : +/- {DEFAULT_QUARTER_CAR_PARAMS.force_max:.0f} N | Slew: {DEFAULT_QUARTER_CAR_PARAMS.slew_rate_max/1000:.1f} kN/s")
    print(f"  Safety Bounds   : Stroke +/- {DEFAULT_QUARTER_CAR_PARAMS.stroke_max*1000:.0f} mm | Dynamic Tire Ground Contact F_z >= 0 N")
    print("-" * 78)


def render_ascii_chart(times: Sequence[float], passive: Sequence[float], active: Sequence[float], height: int = 10, width: int = 60) -> None:
    """Render compact comparative time-series in ASCII."""
    print("\n  [Sprung Mass Vertical Acceleration: Passive (·) vs Active (*)]")
    
    # Downsample to width
    step = max(1, len(times) // width)
    t_sub = [times[i] for i in range(0, len(times), step)][:width]
    p_sub = [passive[i] for i in range(0, len(passive), step)][:width]
    a_sub = [active[i] for i in range(0, len(active), step)][:width]

    all_vals = p_sub + a_sub
    v_min, v_max = min(all_vals), max(all_vals)
    span = max(1e-4, v_max - v_min)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for col in range(len(p_sub)):
        # Passive row
        r_p = int((height - 1) * (1.0 - (p_sub[col] - v_min) / span))
        r_p = max(0, min(height - 1, r_p))
        grid[r_p][col] = "·"

        # Active row
        r_a = int((height - 1) * (1.0 - (a_sub[col] - v_min) / span))
        r_a = max(0, min(height - 1, r_a))
        grid[r_a][col] = "*"

    for row in range(height):
        val = v_max - row * (span / (height - 1))
        line = "".join(grid[row])
        print(f"  {val:+6.2f} m/s^2 |{line}|")
    print(f"         Time |{'^' * width}| {times[-1]:.2f} s\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chassis Active Suspension MPC & CBF Demonstrator")
    parser.add_argument("--road", choices=["A", "B", "C", "D", "bump", "pothole"], default="A", help="Road profile type")
    parser.add_argument("--velocity", type=float, default=72.0, help="Forward vehicle velocity in km/h")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration in seconds")
    parser.add_argument("--benchmark-latency", action="store_true", help="Run high-resolution timing benchmark")
    args = parser.parse_args()

    print_banner()

    vel_ms = args.velocity / 3.6
    benchmarker = SuspensionBenchmarker(dt=0.001, ctrl_dt=0.010)

    road_class_map = {
        "A": RoadClass.CLASS_A,
        "B": RoadClass.CLASS_B,
        "C": RoadClass.CLASS_C,
        "D": RoadClass.CLASS_D,
    }

    if args.road in road_class_map:
        rc = road_class_map[args.road]
        print(f"  Excitation      : ISO 8608 Class {args.road} @ {args.velocity:.1f} km/h ({vel_ms:.1f} m/s)")
        print(f"  Duration        : {args.duration:.1f} s ({int(args.duration*1000):,} dynamic steps)")
        print("-" * 78)

        t0 = time.perf_counter()
        results = benchmarker.benchmark_iso_road(road_class=rc, velocity=vel_ms, duration=args.duration)
        elapsed = time.perf_counter() - t0

        p = results["passive"]
        s = results["skyhook"]
        a = results["active_mpc_cbf"]

        print(f"  Sim Elapsed     : {elapsed*1000:.1f} ms\n")
        print("  PERFORMANCE BENCHMARK MATRIX:")
        print("  " + "-" * 74)
        print(f"  {'Metric':<30} | {'Passive':<11} | {'Skyhook':<11} | {'Active MPC+CBF':<14}")
        print("  " + "-" * 74)
        print(f"  {'RMS Body Accel [m/s^2]':<30} | {p.rms_sprung_accel:<11.4f} | {s.rms_sprung_accel:<11.4f} | {a.rms_sprung_accel:<14.4f}")
        print(f"  {'Comfort Improvement [%]':<30} | {'0.00%':<11} | {s.comfort_improvement_pct:<10.2f}% | {a.comfort_improvement_pct:<13.2f}%")
        print(f"  {'Peak Accel [m/s^2]':<30} | {p.peak_sprung_accel:<11.4f} | {s.peak_sprung_accel:<11.4f} | {a.peak_sprung_accel:<14.4f}")
        print(f"  {'Max Stroke [mm] (<= 50 mm)':<30} | {p.max_suspension_stroke*1000:<11.2f} | {s.max_suspension_stroke*1000:<11.2f} | {a.max_suspension_stroke*1000:<14.2f}")
        print(f"  {'Min Tire Normal Force [N] (>= 0)':<30} | {p.min_tire_normal_force:<11.1f} | {s.min_tire_normal_force:<11.1f} | {a.min_tire_normal_force:<14.1f}")
        print(f"  {'Actuator Energy [kJ]':<30} | {'0.00':<11} | {s.control_energy/1000:<11.2f} | {a.control_energy/1000:<14.2f}")
        print("  " + "-" * 74)

        if a.comfort_improvement_pct >= 25.0:
            print("  >> STATUS: RIDE COMFORT TARGET ACHIEVED (>= 25% RMS REDUCTION) <<")
        else:
            print(f"  >> STATUS: Comfort improvement achieved: {a.comfort_improvement_pct:.2f}% <<")

        if a.max_suspension_stroke <= 0.050 and a.min_tire_normal_force >= 0.0:
            print("  >> SAFETY INVARIANTS: STRICT BOUNDARY PRESERVATION VERIFIED <<")

    if args.benchmark_latency:
        print("\n" + "=" * 78)
        print("  REAL-TIME EMBEDDED LATENCY PROFILER (1,000 CYCLES)")
        print("=" * 78)
        lat_bench = LatencyBenchmarker()
        lat_results = lat_bench.run_benchmark(n_cycles=1000)

        print(f"  {'Software Stage':<25} | {'p50 (Median)':<12} | {'p95':<10} | {'p99':<10} | {'Max':<10}")
        print("  " + "-" * 74)
        for stage, p_data in lat_results.items():
            name = {
                "mpc": "Level 1: MPC Planner",
                "cbf": "Level 2: HOCBF Shield",
                "can": "Level 4: CAN DBC Encode",
                "total_e2e": "TOTAL END-TO-END",
            }.get(stage, stage)
            print(f"  {name:<25} | {p_data.p50_us:<10.1f} us | {p_data.p95_us:<8.1f} us | {p_data.p99_us:<8.1f} us | {p_data.max_us:<8.1f} us")
        print("  " + "-" * 74)
        total_p50 = lat_results["total_e2e"].p50_us
        print(f"  Total Execution Budget: 1,000.0 us | Observed Median: {total_p50:.1f} us ({total_p50/10:.1f}% CPU load at 100 Hz)")
        print("=" * 78)


if __name__ == "__main__":
    main()
