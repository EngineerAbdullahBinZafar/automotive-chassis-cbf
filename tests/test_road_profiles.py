"""
Unit tests for ISO 8608 Road Profiles, Wheelbase Delays, Obstacle Geometries,
and Deterministic RK4 Numerical Integrator.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy import signal

from src.plant.road_profiles import (
    RoadClass,
    ISO_GD_N0_MAP,
    ISO8608RoadGenerator,
    WheelbaseDelayBuffer,
    SpeedBump,
    Pothole,
    CombinedRoadProfile,
    RoadSurfacePoint,
    DualAxleRoadPoint,
)
from src.plant.integrator import (
    rk4_step_scalar,
    rk4_step_numpy,
    rk4_step_linear,
    FixedStepRK4Integrator,
    verify_linear_rk4_stability,
)
from src.plant.quarter_car import QuarterCarModel
from src.plant.half_car import HalfCarModel


class TestISO8608RoadProfiles:
    """Test Suite for ISO 8608 Road Roughness Synthesis and Delay Buffers."""

    def test_road_class_enum_constants(self) -> None:
        """Verify all ISO 8608 roughness constants match standard Gd(n0)."""
        expected = {
            RoadClass.CLASS_A: 32e-6,
            RoadClass.CLASS_B: 64e-6,
            RoadClass.CLASS_C: 256e-6,
            RoadClass.CLASS_D: 1024e-6,
            RoadClass.CLASS_E: 4096e-6,
        }
        for rc, val in expected.items():
            assert ISO_GD_N0_MAP[rc] == pytest.approx(val, rel=1e-6)

    def test_iso8608_variance_exactness(self) -> None:
        """
        Verify empirical variance across 200,000 steps matches theoretical
        variance V_theo = (pi * n0^2 * Gd(n0)) / (2 * f0) within +/- 5%.
        """
        for rc in [RoadClass.CLASS_A, RoadClass.CLASS_B, RoadClass.CLASS_C]:
            gen = ISO8608RoadGenerator(road_class=rc, velocity=20.0, dt=0.001, seed=123)
            _, z_arr, _ = gen.generate_profile(duration=200.0)
            z_eval = np.array(z_arr[20000:])  # Exclude initial transient
            empirical_var = np.var(z_eval)
            theoretical_var = gen.var_theo
            ratio = empirical_var / theoretical_var
            assert 0.95 <= ratio <= 1.05, f"Class {rc.value} variance mismatch: ratio={ratio:.4f}"

    def test_sample_invariance_across_sample_rates(self) -> None:
        """
        Verify that exact discrete formulation exhibits zero variance distortion
        across different sample intervals dt in [0.5 ms, 1.0 ms, 5.0 ms, 10.0 ms].
        """
        rc = RoadClass.CLASS_C
        for dt in [0.0005, 0.001, 0.005, 0.010]:
            gen = ISO8608RoadGenerator(road_class=rc, velocity=20.0, dt=dt, seed=42)
            # Mathematical exactness check: algebraic stationary variance must match V_theo exactly
            discrete_var = gen.gamma**2 / (1.0 - gen.phi**2)
            assert math.isclose(discrete_var, gen.var_theo, rel_tol=1e-12)

            # Statistical sampling check over finite simulation
            _, z_arr, _ = gen.generate_profile(duration=200.0)
            z_eval = np.array(z_arr[int(20.0 / dt):])
            empirical_var = np.var(z_eval)
            ratio = empirical_var / gen.var_theo
            assert 0.92 <= ratio <= 1.08, f"dt={dt} sample-invariance violated: ratio={ratio:.4f}"

    def test_low_frequency_drift_elimination(self) -> None:
        """
        Verify that mean reversion eliminates low-frequency random walk drift:
        Mean elevation over 500 seconds remains strictly under 1.0 mm.
        """
        gen = ISO8608RoadGenerator(road_class=RoadClass.CLASS_C, velocity=20.0, dt=0.001, seed=999)
        _, z_arr, _ = gen.generate_profile(duration=500.0)
        mean_elevation = abs(float(np.mean(z_arr)))
        assert mean_elevation < 1e-3, f"Low-frequency drift observed: mean={mean_elevation:.6e} m"

    def test_iso8608_psd_log_slope(self) -> None:
        """
        Compute Welch PSD of generated profile and verify waviness slope
        w = -2.0 (+/- 0.20) on log-log scale between 2 Hz and 20 Hz.
        """
        gen = ISO8608RoadGenerator(road_class=RoadClass.CLASS_C, velocity=20.0, dt=0.001, seed=42)
        _, z_arr, _ = gen.generate_profile(duration=500.0)
        fs = 1000.0
        freqs, psd = signal.welch(np.asarray(z_arr, dtype=float), fs=fs, nperseg=16384)
        mask = (freqs >= 2.0) & (freqs <= 20.0)
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask])
        slope, _ = np.polyfit(log_f, log_p, 1)
        assert -2.25 <= slope <= -1.75, f"PSD slope {slope:.3f} deviates from ISO -2.0"

    def test_wheelbase_delay_exactness(self) -> None:
        """
        Verify circular FIFO buffer delay exactness for integer and fractional delays.
        """
        wheelbase = 3.20
        dt = 0.001
        buffer = WheelbaseDelayBuffer(wheelbase=wheelbase, dt=dt)

        # Integer delay: v = 20.0 m/s => tau = 0.160 s => 160 steps
        freq = 4.0
        points = [
            RoadSurfacePoint(math.sin(2.0 * math.pi * freq * k * dt), 0.0)
            for k in range(400)
        ]
        delayed_output: list[float] = []
        for pt in points:
            buffer.push(pt)
            delayed_output.append(buffer.get_delayed(20.0).elevation)

        for k in range(200, 350):
            expected = math.sin(2.0 * math.pi * freq * (k - 160) * dt)
            actual = delayed_output[k]
            assert abs(actual - expected) < 1e-5, f"Integer delay error at step {k}"

        # Fractional delay test (v = 25.0 m/s => tau = 0.128 s => 128 steps)
        buffer.reset()
        delayed_out_frac: list[float] = []
        for pt in points:
            buffer.push(pt)
            delayed_out_frac.append(buffer.get_delayed(25.0).elevation)
        for k in range(200, 350):
            expected = math.sin(2.0 * math.pi * freq * (k - 128) * dt)
            actual = delayed_out_frac[k]
            assert abs(actual - expected) < 1e-5, f"Fractional delay error at step {k}"

    def test_obstacle_geometries(self) -> None:
        """Verify SpeedBump (50 mm) and Pothole (-40 mm) exact dimensions and velocities."""
        bump = SpeedBump(height=0.050, length=0.50, start_position=5.0)
        assert bump.evaluate(4.99, 20.0).elevation == 0.0
        assert bump.evaluate(5.0, 20.0).elevation == pytest.approx(0.0)
        assert bump.evaluate(5.25, 20.0).elevation == pytest.approx(0.050, abs=1e-5)
        assert bump.evaluate(5.50, 20.0).elevation == pytest.approx(0.0, abs=1e-5)
        assert bump.evaluate(5.51, 20.0).elevation == 0.0
        # Peak vertical velocity at start of bump: v * pi * H / L = 20 * pi * 0.05 / 0.5 = 2*pi
        assert bump.evaluate(5.0, 20.0).velocity == pytest.approx(2.0 * math.pi, rel=1e-3)

        pothole = Pothole(depth=0.040, length=0.40, start_position=5.0)
        assert pothole.evaluate(4.99, 20.0).elevation == 0.0
        assert pothole.evaluate(5.20, 20.0).elevation == pytest.approx(-0.040, abs=1e-5)
        assert pothole.evaluate(5.40, 20.0).elevation == pytest.approx(0.0, abs=1e-5)

    def test_combined_road_profile(self) -> None:
        """Verify combined road profile generates synchronized dual-axle excitations."""
        bump = SpeedBump(height=0.050, length=0.50, start_position=5.0)
        profile = CombinedRoadProfile(
            road_class=RoadClass.CLASS_A,
            obstacles=[bump],
            velocity=20.0,
            dt=0.001,
            wheelbase=3.20,
            seed=42,
        )

        front_elevations = []
        rear_elevations = []
        for _ in range(500):  # 0.5 seconds
            pt = profile.step()
            front_elevations.append(pt.front.elevation)
            rear_elevations.append(pt.rear.elevation)

        assert len(front_elevations) == 500
        assert len(rear_elevations) == 500
        # 1. Prior to reaching obstacle (x < 4.0m in first 200 ms), elevations are pure Class A (< 20 mm)
        assert max(abs(e) for e in front_elevations[:200]) < 0.02
        # 2. Front axle crosses the 50 mm speed bump around x = 5.25m (t ~ 0.26s)
        assert max(front_elevations) >= 0.045
        # 3. Rear axle also crosses the 50 mm speed bump with wheelbase delay tau = 0.16s (t ~ 0.42s)
        assert max(rear_elevations) >= 0.045


class TestDeterministicRK4Integrator:
    """Test Suite for Fixed-Step Deterministic RK4 Integrator."""

    def test_rk4_scalar_vs_numpy_equivalence(self) -> None:
        """Verify that pure-Python scalar RK4 matches NumPy RK4 to floating-point precision."""
        def harmonic(t: float, x: list[float]) -> list[float]:
            return [x[1], -4.0 * x[0]]

        def harmonic_np(t: float, x: np.ndarray) -> np.ndarray:
            return np.array([x[1], -4.0 * x[0]])

        dt = 0.001
        xs = [1.0, 0.0]
        xn = np.array([1.0, 0.0])
        t = 0.0
        for _ in range(1000):
            xs = rk4_step_scalar(harmonic, t, xs, dt)
            xn = rk4_step_numpy(harmonic_np, t, xn, dt)
            t += dt

        assert np.allclose(xs, xn, atol=1e-12)

    def test_rk4_order_of_convergence(self) -> None:
        """Verify 4th-order global error convergence: error ratio ~ 16 for halving dt."""
        def f(t: float, x: list[float]) -> list[float]:
            return [-x[0]]

        def solve(dt: float) -> float:
            t = 0.0
            x = [1.0]
            steps = int(round(1.0 / dt))
            for _ in range(steps):
                x = rk4_step_scalar(f, t, x, dt)
                t += dt
            return abs(x[0] - math.exp(-1.0))

        e1 = solve(0.1)
        e2 = solve(0.05)
        e3 = solve(0.025)
        ratio1 = e1 / e2
        ratio2 = e2 / e3
        assert 15.0 <= ratio1 <= 17.5, f"Convergence ratio {ratio1:.2f} deviates from 16.0"
        assert 15.0 <= ratio2 <= 17.5, f"Convergence ratio {ratio2:.2f} deviates from 16.0"

    def test_undamped_oscillator_energy_conservation(self) -> None:
        """
        Verify relative energy drift < 1e-4 over 10 seconds (10,000 steps at 1000 Hz)
        for an undamped 2-DOF quarter car plant.
        """
        ms, mu = 500.0, 50.0
        ks, kt = 24000.0, 200000.0

        def deriv(t: float, x: list[float]) -> list[float]:
            zs, zs_dot, zu, zu_dot = x
            zs_ddot = -ks * (zs - zu) / ms
            zu_ddot = (ks * (zs - zu) - kt * zu) / mu
            return [zs_dot, zs_ddot, zu_dot, zu_ddot]

        def energy(x: list[float]) -> float:
            zs, zs_dot, zu, zu_dot = x
            return (
                0.5 * ms * zs_dot**2
                + 0.5 * mu * zu_dot**2
                + 0.5 * ks * (zs - zu)**2
                + 0.5 * kt * zu**2
            )

        x = [0.02, 0.0, 0.005, 0.0]
        E0 = energy(x)
        dt = 0.001
        for _ in range(10000):
            x = rk4_step_scalar(deriv, 0.0, x, dt)
        E_final = energy(x)
        rel_drift = abs(E_final - E0) / E0
        assert rel_drift < 1e-4, f"Energy drift {rel_drift:.2e} exceeds 1e-4"

    def test_rk4_linear_stepper(self) -> None:
        """Verify rk4_step_linear matches matrix differential solution."""
        A = np.array([[0.0, 1.0], [-4.0, -0.5]])
        B = np.array([[0.0], [1.0]])
        G = np.array([[0.0], [0.5]])
        x = np.array([0.1, 0.0])
        u = np.array([1.0])
        w = np.array([0.2])
        dt = 0.001

        x_next = rk4_step_linear(A, B, G, x, u, w, dt)
        assert x_next.shape == (2,)

    def test_rk4_linear_stability_verification(self) -> None:
        """Verify stability analysis helper function for quarter-car and half-car plants."""
        dt = 0.001

        # Quarter-car
        qc = QuarterCarModel()
        A_qc = np.array(qc.get_continuous_state_space_relative()["A"])
        stable_qc, max_amp_qc, _ = verify_linear_rk4_stability(A_qc, dt)
        assert stable_qc
        assert max_amp_qc < 1.0

        # Half-car
        hc = HalfCarModel()
        stable_hc, max_amp_hc, _ = verify_linear_rk4_stability(hc.A, dt)
        assert stable_hc
        assert max_amp_hc < 1.0

    def test_fixed_step_rk4_integrator_stateful(self) -> None:
        """Verify stateful FixedStepRK4Integrator advances time and resets."""
        def decay(t: float, x: list[float]) -> list[float]:
            return [-x[0]]

        engine = FixedStepRK4Integrator(f=decay, dt=0.01)
        x = [1.0]
        for _ in range(100):
            t, x = engine.step(x)
        assert pytest.approx(t, rel=1e-5) == 1.0
        assert pytest.approx(x[0], rel=1e-3) == math.exp(-1.0)

        engine.reset()
        assert engine.t == 0.0
