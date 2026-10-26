"""
Comprehensive unit tests for Quarter-Car and Half-Car Vehicle Dynamics Plant Models.
BMW Group / Rolls-Royce Executive Luxury Sedan Chassis Benchmark.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from src.plant.parameters import (
    QuarterCarParameters,
    DEFAULT_QUARTER_CAR_PARAMS,
    HalfCarParameters,
    DEFAULT_HALF_CAR_PARAMS,
)
from src.plant.quarter_car import (
    QuarterCarModel,
    QuarterCarState,
    ConstraintStatus,
)
from src.plant.half_car import (
    HalfCarModel,
    HalfCarState,
)


# =====================================================================
# SECTION 1: Quarter-Car Plant Model Unit Tests
# =====================================================================

class TestQuarterCarDynamics:
    """Test Suite for 2-DOF Quarter-Car Dynamics and Executive Sedan Calibration."""

    def test_quarter_car_parameters_default(self) -> None:
        """Verify default parameters match luxury sedan benchmark."""
        p = DEFAULT_QUARTER_CAR_PARAMS
        assert p.ms == 500.0
        assert p.mu == 50.0
        assert p.ks == 24000.0
        assert p.cs == 2000.0
        assert p.kt == 200000.0
        assert p.ct == 0.0
        assert p.g == 9.81
        assert p.stroke_min == -0.050
        assert p.stroke_max == 0.050
        assert p.force_min == -2500.0
        assert p.force_max == 2500.0
        assert p.slew_rate_max == 15000.0
        assert p.total_mass == 550.0
        assert pytest.approx(p.f_z_static, rel=1e-5) == 5395.5
        assert pytest.approx(p.sprung_preload, rel=1e-5) == 4905.0
        assert pytest.approx(p.static_tire_deflection, rel=1e-5) == 0.0269775
        assert pytest.approx(p.static_suspension_deflection, rel=1e-5) == 0.204375

    def test_quarter_car_parameter_validation(self) -> None:
        """Defensive validation: non-positive mass/stiffness or invalid bounds must raise ValueError."""
        with pytest.raises(ValueError, match="Masses must be strictly positive"):
            QuarterCarParameters(ms=-10.0)
        with pytest.raises(ValueError, match="Masses must be strictly positive"):
            QuarterCarParameters(mu=0.0)
        with pytest.raises(ValueError, match="Stiffnesses must be strictly positive"):
            QuarterCarParameters(ks=-500.0)
        with pytest.raises(ValueError, match="Damping must be non-negative"):
            QuarterCarParameters(cs=-1.0)
        with pytest.raises(ValueError, match="stroke_min"):
            QuarterCarParameters(stroke_min=0.06, stroke_max=0.05)
        with pytest.raises(ValueError, match="force_min"):
            QuarterCarParameters(force_min=3000.0, force_max=2500.0)

    def test_quarter_car_eigenvalues_bounce_and_hop(self) -> None:
        """Verify continuous poles: body bounce ~1.057 Hz, wheel hop ~10.498 Hz."""
        p = DEFAULT_QUARTER_CAR_PARAMS
        model = QuarterCarModel(p)
        ss = model.get_continuous_state_space_relative()
        A = np.array(ss["A"])

        eigvals = np.linalg.eigvals(A)
        freqs = np.sort(np.abs(eigvals) / (2.0 * np.pi))
        zetas = -np.real(eigvals) / np.abs(eigvals)
        zetas = np.sort(zetas)

        # Bounce mode (lowest frequency pair)
        assert pytest.approx(freqs[0], abs=0.02) == 1.057
        assert pytest.approx(zetas[0], abs=0.02) == 0.246

        # Wheel hop mode (highest frequency pair)
        assert pytest.approx(freqs[2], abs=0.1) == 10.498
        assert pytest.approx(zetas[2], abs=0.02) == 0.309

    def test_quarter_car_matrix_equivalence(self) -> None:
        """Characteristic polynomials of relative and absolute state-spaces must be identical."""
        model = QuarterCarModel()
        A_rel = np.array(model.get_continuous_state_space_relative()["A"])
        A_abs = np.array(model.get_continuous_state_space_absolute()["A"])

        e_rel = np.sort(np.abs(np.linalg.eigvals(A_rel)))
        e_abs = np.sort(np.abs(np.linalg.eigvals(A_abs)))
        assert np.allclose(e_rel, e_abs, atol=1e-10)

    def test_quarter_car_derivatives_equilibrium(self) -> None:
        """At zero state with zero inputs, all derivatives must evaluate to identically 0.0."""
        model = QuarterCarModel()
        d_abs = model.derivatives_absolute((0.0, 0.0, 0.0, 0.0), 0.0, 0.0)
        assert d_abs == (0.0, 0.0, 0.0, 0.0)

        d_rel = model.derivatives_relative((0.0, 0.0, 0.0, 0.0), 0.0, 0.0)
        assert d_rel == (0.0, 0.0, 0.0, 0.0)

    def test_quarter_car_active_force_acceleration(self) -> None:
        """Applying Fa = 1000 N at rest produces ddzs = 2.0 m/s^2 and ddzu = -20.0 m/s^2."""
        model = QuarterCarModel()
        d = model.derivatives_absolute((0.0, 0.0, 0.0, 0.0), 1000.0, 0.0)
        assert pytest.approx(d[1], rel=1e-6) == 2.0   # 1000 / 500 kg
        assert pytest.approx(d[3], rel=1e-6) == -20.0 # -1000 / 50 kg

    def test_quarter_car_energy_conservation_undamped(self) -> None:
        """Undamped system (cs=0, ct=0) must conserve mechanical energy with rel error < 1e-5 over 1000 steps."""
        p = QuarterCarParameters(cs=0.0, ct=0.0)
        model = QuarterCarModel(params=p, initial_zs=0.02, initial_zu=0.005)

        def total_energy(zs: float, dzs: float, zu: float, dzu: float) -> float:
            return (
                0.5 * p.ms * dzs**2
                + 0.5 * p.mu * dzu**2
                + 0.5 * p.ks * (zs - zu)**2
                + 0.5 * p.kt * zu**2
            )

        e0 = total_energy(model._zs, model._dzs, model._zu, model._dzu)
        dt = 0.001
        for _ in range(1000):
            model.step(0.0, dt, 0.0)

        ef = total_energy(model._zs, model._dzs, model._zu, model._dzu)
        rel_error = abs(ef - e0) / e0
        assert rel_error < 1e-5

    def test_quarter_car_step_response_settling(self) -> None:
        """Step road excitation (50 mm) settles to zs(inf) = 50 mm, zu(inf) = 50 mm, overshoot ~53%."""
        model = QuarterCarModel()
        dt = 0.001
        zr = 0.050  # 50 mm road step
        max_zs = 0.0
        for _ in range(4000):  # 4.0 seconds
            st = model.step_passive(dt, zr)
            if model._zs > max_zs:
                max_zs = model._zs

        assert pytest.approx(model._zs, abs=1e-3) == 0.050
        assert pytest.approx(model._zu, abs=1e-3) == 0.050
        assert pytest.approx(model._dzs, abs=2e-3) == 0.0
        overshoot = (max_zs - zr) / zr * 100.0
        assert 50.0 <= overshoot <= 56.0

    def test_quarter_car_constraints(self) -> None:
        """Verify stroke margin, stroke violation, tire liftoff, and actuator saturation flags."""
        model = QuarterCarModel()

        # Nominal inside limits
        c1 = model.check_constraints(QuarterCarState(0.02, 0.0, 0.005, 0.0, 0.0), 0.0)
        assert not c1.stroke_violation
        assert not c1.tire_liftoff
        assert c1.stroke_margin > 0.0
        assert not c1.actuator_saturated

        # Stroke violation (|zs - zu| > 0.050 m)
        c2 = model.check_constraints(QuarterCarState(0.055, 0.0, 0.0, 0.0, 0.0), 0.0)
        assert c2.stroke_violation
        assert c2.stroke_margin < 0.0

        # Tire liftoff (extension > static deflection 0.02698 m)
        c3 = model.check_constraints(QuarterCarState(0.0, 0.0, 0.030, 0.0, 0.0), 0.0)
        assert c3.tire_liftoff
        assert c3.tire_force_total == 0.0

        # Actuator saturation
        c4 = model.check_constraints(QuarterCarState(0.0, 0.0, 0.0, 0.0, 0.0), 2500.0)
        assert c4.actuator_saturated

    def test_quarter_car_passive_mode_equivalence(self) -> None:
        """step_passive(dt, zr) must match step(0.0, dt, zr) exactly."""
        m1 = QuarterCarModel()
        m2 = QuarterCarModel()
        dt = 0.001
        for step_idx in range(100):
            zr = 0.01 * math.sin(step_idx * 0.1)
            s1 = m1.step(0.0, dt, zr)
            s2 = m2.step_passive(dt, zr)
            assert s1 == s2

    def test_quarter_car_realtime_step_latency(self) -> None:
        """Verify single step simulation executes in < 100 microseconds (well under 1.0 ms requirement)."""
        model = QuarterCarModel()
        n_steps = 1000
        t0 = time.perf_counter()
        for _ in range(n_steps):
            model.step(100.0, 0.001, 0.01)
        t1 = time.perf_counter()
        mean_latency_us = (t1 - t0) / n_steps * 1e6
        assert mean_latency_us < 100.0, f"Mean step latency {mean_latency_us:.2f} us exceeds 100 us"


# =====================================================================
# SECTION 2: Half-Car 4-DOF Pitch-Bounce Unit Tests
# =====================================================================

class TestHalfCarDynamics:
    """Test Suite for 4-DOF Half-Car Pitch-Bounce Dynamics and Olley Isolation."""

    def test_half_car_parameters_consistency(self) -> None:
        """Verify parameters, static axle loads, and dynamic index."""
        p = DEFAULT_HALF_CAR_PARAMS
        assert p.ms == 1000.0
        assert p.iy == 1500.0
        assert p.a == 1.55
        assert p.b == 1.65
        assert p.l == 3.20
        assert p.muf == 52.0
        assert p.mur == 48.0
        assert p.ksf == 26000.0
        assert p.ksr == 24000.0
        assert p.csf == 2100.0
        assert p.csr == 1900.0
        assert p.ktf == 200000.0
        assert p.ktr == 200000.0

        # Normal loads
        assert pytest.approx(p.f_zf_static, abs=0.01) == 5568.40
        assert pytest.approx(p.f_zr_static, abs=0.01) == 5222.60
        assert pytest.approx(p.f_zf_static + p.f_zr_static, abs=0.01) == (p.ms + p.muf + p.mur) * p.g

        # Olley Dynamic Index: Iy / (ms * a * b) = 1500 / (1000 * 1.55 * 1.65) = 0.5865
        assert pytest.approx(p.dynamic_index, abs=1e-3) == 0.5865

    def test_half_car_state_space_dimensions_and_zeros(self) -> None:
        """Verify A is 8x8, B is 8x2, G is 8x2 with strict zero and identity kinematic rows."""
        model = HalfCarModel()
        A = model.A
        B = model.B
        G = model.G

        assert A.shape == (8, 8)
        assert B.shape == (8, 2)
        assert G.shape == (8, 2)

        # Kinematic identity rows: dx0/dt = x1, dx2/dt = x3, dx4/dt = x5, dx6/dt = x7
        assert A[0, 1] == 1.0 and np.sum(np.abs(A[0, :])) == 1.0
        assert A[2, 3] == 1.0 and np.sum(np.abs(A[2, :])) == 1.0
        assert A[4, 5] == 1.0 and np.sum(np.abs(A[4, :])) == 1.0
        assert A[6, 7] == 1.0 and np.sum(np.abs(A[6, :])) == 1.0

    def test_half_car_asymptotic_stability(self) -> None:
        """All 8 continuous eigenvalues of A must have strictly negative real parts (Hurwitz)."""
        model = HalfCarModel()
        eigvals = np.linalg.eigvals(model.A)
        assert len(eigvals) == 8
        assert np.all(np.real(eigvals) < 0.0), "System matrix A is not strictly Hurwitz stable"

    def test_maurice_olley_pitch_bounce_ratio(self) -> None:
        """
        Maurice Olley ride-comfort criterion:
        f_pitch / f_bounce = 1.421 / 1.076 = 1.32 > 1.0 (recommended 1.15 - 1.35).
        """
        model = HalfCarModel()
        eigvals = np.linalg.eigvals(model.A)
        fns = np.sort(np.abs(eigvals) / (2.0 * math.pi))

        # Modal frequencies
        f_bounce = fns[0]
        f_pitch = fns[2]
        ratio = f_pitch / f_bounce

        assert pytest.approx(f_bounce, abs=0.03) == 1.076
        assert pytest.approx(f_pitch, abs=0.03) == 1.421
        assert 1.28 <= ratio <= 1.35
        assert ratio > 1.0

    def test_half_car_wheel_hop_frequencies(self) -> None:
        """Verify front and rear wheel hop natural frequencies are ~10.68 Hz and ~10.26 Hz."""
        model = HalfCarModel()
        eigvals = np.linalg.eigvals(model.A)
        fns = np.sort(np.abs(eigvals) / (2.0 * math.pi))

        f_hop_rear = fns[4]
        f_hop_front = fns[6]

        assert pytest.approx(f_hop_rear, abs=0.25) == 10.265
        assert pytest.approx(f_hop_front, abs=0.25) == 10.676

    def test_modal_decoupling_moments(self) -> None:
        """
        Verify static stiffness coupling |a*ksf - b*ksr| is < 2% of front moment
        and damping coupling |a*csf - b*csr| is < 4% of front moment.
        """
        p = DEFAULT_HALF_CAR_PARAMS
        k_coup = abs(p.stiffness_coupling) / (p.a * p.ksf)
        c_coup = abs(p.damping_coupling) / (p.a * p.csf)

        assert abs(p.stiffness_coupling) == 700.0
        assert abs(p.damping_coupling) == 120.0
        assert k_coup < 0.02
        assert c_coup < 0.04

    def test_pure_heave_symmetric_response(self) -> None:
        """Initial heave (zs = 20 mm, theta = 0) produces negligible cross-talk pitch (|theta| < 1e-4 rad)."""
        model = HalfCarModel()
        model.reset(HalfCarState(zs=0.02, zs_dot=0.0, theta=0.0, theta_dot=0.0,
                                 zuf=0.0, zuf_dot=0.0, zur=0.0, zur_dot=0.0))
        max_theta = 0.0
        dt = 0.001
        for _ in range(2000):  # 2.0 seconds
            st = model.step_passive(dt, (0.0, 0.0))
            if abs(st.theta) > max_theta:
                max_theta = abs(st.theta)

        assert max_theta < 1e-4, f"Cross-talk pitch {max_theta:.6e} rad exceeds 1e-4 rad limit"

    def test_pure_pitch_asymmetric_response(self) -> None:
        """Initial pitch (theta = 0.01 rad, zs = 0) produces negligible cross-talk heave (|zs| < 1e-4 m)."""
        model = HalfCarModel()
        model.reset(HalfCarState(zs=0.0, zs_dot=0.0, theta=0.01, theta_dot=0.0,
                                 zuf=0.0, zuf_dot=0.0, zur=0.0, zur_dot=0.0))
        max_zs = 0.0
        dt = 0.001
        for _ in range(2000):  # 2.0 seconds
            st = model.step_passive(dt, (0.0, 0.0))
            if abs(st.zs) > max_zs:
                max_zs = abs(st.zs)

        assert max_zs < 1e-4, f"Cross-talk heave {max_zs:.6e} m exceeds 1e-4 m limit"

    def test_kinematic_attachment_points(self) -> None:
        """Verify attachment displacements: z_sf = zs - a*theta, z_sr = zs + b*theta."""
        p = DEFAULT_HALF_CAR_PARAMS
        zs = 0.020
        theta = 0.005  # nose-down
        z_sf_expected = zs - p.a * theta  # 0.020 - 1.55*0.005 = 0.01225 m
        z_sr_expected = zs + p.b * theta  # 0.020 + 1.65*0.005 = 0.02825 m

        model = HalfCarModel()
        st = model.step((0.0, 0.0), 0.001, (0.0, 0.0))
        # Feed explicit state check
        st_test = HalfCarState(zs=zs, zs_dot=0.0, theta=theta, theta_dot=0.0,
                               zuf=0.0, zuf_dot=0.0, zur=0.0, zur_dot=0.0)
        model.reset(st_test)
        st_out = model.step((0.0, 0.0), 0.001, (0.0, 0.0))
        assert pytest.approx(st_out.z_sf, abs=1e-4) == z_sf_expected
        assert pytest.approx(st_out.z_sr, abs=1e-4) == z_sr_expected

    def test_matrix_vs_scalar_derivative_equivalence(self) -> None:
        """Scalar Newton-Euler derivatives must match matrix product A*x + B*u + G*w within 1e-12."""
        model = HalfCarModel()
        rng = np.random.default_rng(123)

        for _ in range(50):
            x = rng.uniform(-0.02, 0.02, size=8)
            u = (float(rng.uniform(-1000, 1000)), float(rng.uniform(-1000, 1000)))
            w = (float(rng.uniform(-0.02, 0.02)), float(rng.uniform(-0.02, 0.02)))

            d_scalar = model.derivatives(x.tolist(), u, w)
            d_matrix = model.A @ x + model.B @ np.array(u) + model.G @ np.array(w)

            assert np.allclose(d_scalar, d_matrix, atol=1e-12)

    def test_actuator_force_modal_isolation(self) -> None:
        """
        Symmetric force (+1 kN, +1 kN) drives pure vertical heave (ddzs = 2.0 m/s^2).
        Differential force (+1 kN, -1 kN) drives pure pitch acceleration (ddzs = 0.0 m/s^2).
        """
        model = HalfCarModel()
        x0 = [0.0] * 8
        w0 = (0.0, 0.0)

        # 1. Symmetric
        d_sym = model.derivatives(x0, (1000.0, 1000.0), w0)
        assert pytest.approx(d_sym[1], rel=1e-5) == 2.0  # ddzs = 2000 / 1000 kg
        assert pytest.approx(d_sym[3], rel=1e-2) == 0.0667 # near-zero pitch cross-talk (1000*(1.65-1.55)/1500)

        # 2. Anti-symmetric / differential
        d_diff = model.derivatives(x0, (1000.0, -1000.0), w0)
        assert pytest.approx(d_diff[1], abs=1e-6) == 0.0 # ddzs = 0
        assert d_diff[3] < 0.0 # Nose lifts, negative pitch acceleration

    def test_half_car_physical_constraint_monitoring(self) -> None:
        """Verify front/rear stroke violation and tire liftoff detection."""
        model = HalfCarModel()
        # Forced front stroke violation (d_sf = 0.055 m)
        st_stroke = HalfCarState(
            zs=0.055, zs_dot=0.0, theta=0.0, theta_dot=0.0,
            zuf=0.0, zuf_dot=0.0, zur=0.0, zur_dot=0.0,
            susp_defl_front=0.055, susp_defl_rear=0.0,
            tire_defl_front=0.0, tire_defl_rear=0.0,
        )
        c = model.check_constraints(st_stroke)
        assert c["stroke_violation_front"]
        assert not c["stroke_violation_rear"]

        # Forced front tire liftoff (deflection > 0.02784 m)
        st_liftoff = HalfCarState(
            zs=0.0, zs_dot=0.0, theta=0.0, theta_dot=0.0,
            zuf=0.030, zuf_dot=0.0, zur=0.0, zur_dot=0.0,
            susp_defl_front=0.0, susp_defl_rear=0.0,
            tire_defl_front=0.030, tire_defl_rear=0.0,
        )
        c2 = model.check_constraints(st_liftoff)
        assert c2["tire_liftoff_front"]
        assert c2["tire_force_front"] == 0.0

    def test_half_car_step_latency_benchmark(self) -> None:
        """1000 half-car step simulation calls execute in < 200 us/step (well under 1.0 ms requirement)."""
        model = HalfCarModel()
        n_steps = 1000
        t0 = time.perf_counter()
        for _ in range(n_steps):
            model.step((100.0, -100.0), 0.001, (0.01, 0.01))
        t1 = time.perf_counter()
        mean_latency_us = (t1 - t0) / n_steps * 1e6
        assert mean_latency_us < 200.0, f"Mean half-car step latency {mean_latency_us:.2f} us exceeds 200 us"
