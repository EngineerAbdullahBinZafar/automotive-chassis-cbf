"""
Vehicle and suspension physical parameter specifications.
Calibrated for Executive Luxury Sedan benchmark (BMW 7-Series / Rolls-Royce Ghost scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QuarterCarParameters:
    """
    Quarter-car physical parameter specification.
    Executive luxury sedan corner calibrated for 1.057 Hz bounce and 10.5 Hz wheel hop.
    """
    ms: float = 500.0              # Sprung mass [kg]
    mu: float = 50.0               # Unsprung mass [kg]
    ks: float = 24000.0            # Suspension spring stiffness [N/m]
    cs: float = 2000.0             # Suspension damping coefficient [N*s/m]
    kt: float = 200000.0           # Tire vertical stiffness [N/m]
    ct: float = 0.0                # Tire internal carcass damping [N*s/m]
    g: float = 9.81                # Acceleration of gravity [m/s^2]
    stroke_min: float = -0.050     # Lower suspension deflection limit (jounce/compression) [m]
    stroke_max: float = 0.050      # Upper suspension deflection limit (rebound/extension) [m]
    force_min: float = -2500.0     # Actuator maximum negative force [N]
    force_max: float = 2500.0      # Actuator maximum positive force [N]
    slew_rate_max: float = 15000.0 # Actuator force maximum slew rate [N/s]
    tire_safety_factor: float = 0.85 # HOCBF tire contact conservatism factor

    def __post_init__(self) -> None:
        if self.ms <= 0.0 or self.mu <= 0.0:
            raise ValueError(f"Masses must be strictly positive: ms={self.ms}, mu={self.mu}")
        if self.ks <= 0.0 or self.kt <= 0.0:
            raise ValueError(f"Stiffnesses must be strictly positive: ks={self.ks}, kt={self.kt}")
        if self.cs < 0.0 or self.ct < 0.0:
            raise ValueError(f"Damping must be non-negative: cs={self.cs}, ct={self.ct}")
        if self.stroke_min >= self.stroke_max:
            raise ValueError(
                f"stroke_min ({self.stroke_min}) must be strictly less than stroke_max ({self.stroke_max})"
            )
        if self.force_min >= self.force_max:
            raise ValueError(
                f"force_min ({self.force_min}) must be strictly less than force_max ({self.force_max})"
            )

    @property
    def total_mass(self) -> float:
        """Total corner mass (ms + mu) [kg]."""
        return self.ms + self.mu

    @property
    def f_z_static(self) -> float:
        """Static normal ground reaction force (ms + mu) * g [N]."""
        return self.total_mass * self.g

    @property
    def sprung_preload(self) -> float:
        """Static weight carried by suspension spring ms * g [N]."""
        return self.ms * self.g

    @property
    def static_tire_deflection(self) -> float:
        """Static tire compression under total corner mass [m]."""
        return self.f_z_static / self.kt

    @property
    def static_suspension_deflection(self) -> float:
        """Static suspension spring compression under sprung mass [m]."""
        return self.sprung_preload / self.ks

    @property
    def natural_frequencies(self) -> Tuple[float, float]:
        """Analytical uncoupled natural frequencies (bounce_fn, hop_fn) [Hz]."""
        fn_bounce = (1.0 / (2.0 * math.pi)) * math.sqrt(self.ks / self.ms)
        fn_hop = (1.0 / (2.0 * math.pi)) * math.sqrt((self.ks + self.kt) / self.mu)
        return (fn_bounce, fn_hop)


@dataclass(frozen=True)
class HalfCarParameters:
    """
    Half-car (4-DOF pitch-bounce) physical parameter specification.
    Executive luxury sedan calibrated for Maurice Olley pitch-bounce ratio fp/fb = 1.32.
    """
    ms: float = 1000.0             # Half-car sprung mass [kg]
    iy: float = 1500.0             # Pitch moment of inertia [kg*m^2]
    l: float = 3.20                # Wheelbase (a + b) [m]
    a: float = 1.55                # CG to front axle distance [m]
    b: float = 1.65                # CG to rear axle distance [m]
    muf: float = 52.0              # Front unsprung mass [kg]
    mur: float = 48.0              # Rear unsprung mass [kg]
    ksf: float = 26000.0           # Front suspension spring stiffness [N/m]
    ksr: float = 24000.0           # Rear suspension spring stiffness [N/m]
    csf: float = 2100.0            # Front suspension damping coefficient [N*s/m]
    csr: float = 1900.0            # Rear suspension damping coefficient [N*s/m]
    ktf: float = 200000.0          # Front tire vertical stiffness [N/m]
    ktr: float = 200000.0          # Rear tire vertical stiffness [N/m]
    g: float = 9.81                # Acceleration of gravity [m/s^2]
    stroke_min: float = -0.050     # Lower suspension stroke limit [m]
    stroke_max: float = 0.050      # Upper suspension stroke limit [m]
    force_min: float = -2500.0     # Actuator maximum negative force [N]
    force_max: float = 2500.0      # Actuator maximum positive force [N]
    slew_rate_max: float = 15000.0 # Actuator force maximum slew rate [N/s]

    def __post_init__(self) -> None:
        if self.ms <= 0.0 or self.muf <= 0.0 or self.mur <= 0.0:
            raise ValueError("All masses must be strictly positive.")
        if self.iy <= 0.0:
            raise ValueError("Pitch moment of inertia iy must be strictly positive.")
        if self.ksf <= 0.0 or self.ksr <= 0.0 or self.ktf <= 0.0 or self.ktr <= 0.0:
            raise ValueError("All spring stiffnesses must be strictly positive.")
        if self.csf < 0.0 or self.csr < 0.0:
            raise ValueError("Damping coefficients must be non-negative.")
        if self.a <= 0.0 or self.b <= 0.0 or self.l <= 0.0:
            raise ValueError("Wheelbase dimensions must be strictly positive.")
        if abs((self.a + self.b) - self.l) > 1e-4:
            raise ValueError(
                f"Axle distance sum a+b ({self.a + self.b}) must equal wheelbase l ({self.l})"
            )
        if self.stroke_min >= self.stroke_max:
            raise ValueError("stroke_min must be strictly less than stroke_max")
        if self.force_min >= self.force_max:
            raise ValueError("force_min must be strictly less than force_max")

    # Compatibility aliases for alternate notations (m_s, I_y, L, etc.)
    @property
    def m_s(self) -> float:
        return self.ms

    @property
    def I_y(self) -> float:
        return self.iy

    @property
    def L(self) -> float:
        return self.l

    @property
    def m_uf(self) -> float:
        return self.muf

    @property
    def m_ur(self) -> float:
        return self.mur

    @property
    def k_sf(self) -> float:
        return self.ksf

    @property
    def k_sr(self) -> float:
        return self.ksr

    @property
    def c_sf(self) -> float:
        return self.csf

    @property
    def c_sr(self) -> float:
        return self.csr

    @property
    def k_tf(self) -> float:
        return self.ktf

    @property
    def k_tr(self) -> float:
        return self.ktr

    @property
    def stroke_limit(self) -> float:
        return self.stroke_max

    @property
    def f_actuator_max(self) -> float:
        return self.force_max

    @property
    def m_sf(self) -> float:
        """Static sprung mass carried on front axle [kg]."""
        return self.ms * (self.b / self.l)

    @property
    def m_sr(self) -> float:
        """Static sprung mass carried on rear axle [kg]."""
        return self.ms * (self.a / self.l)

    @property
    def total_mass(self) -> float:
        """Total half-car mass (ms + muf + mur) [kg]."""
        return self.ms + self.muf + self.mur

    @property
    def f_zf_static(self) -> float:
        """Static normal ground force on front axle (m_sf + m_uf) * g [N]."""
        return (self.m_sf + self.muf) * self.g

    @property
    def f_zr_static(self) -> float:
        """Static normal ground force on rear axle (m_sr + m_ur) * g [N]."""
        return (self.m_sr + self.mur) * self.g

    @property
    def static_tire_deflection_front(self) -> float:
        """Static front tire deflection [m]."""
        return self.f_zf_static / self.ktf

    @property
    def static_tire_deflection_rear(self) -> float:
        """Static rear tire deflection [m]."""
        return self.f_zr_static / self.ktr

    @property
    def tire_liftoff_dz_f(self) -> float:
        """Front tire deflection causing ground contact loss [m]."""
        return self.f_zf_static / self.ktf

    @property
    def tire_liftoff_dz_r(self) -> float:
        """Rear tire deflection causing ground contact loss [m]."""
        return self.f_zr_static / self.ktr

    @property
    def dynamic_index(self) -> float:
        """Maurice Olley dynamic index: Iy / (ms * a * b)."""
        return self.iy / (self.ms * self.a * self.b)

    @property
    def stiffness_coupling(self) -> float:
        """Static pitch-bounce stiffness coupling moment: a * ksf - b * ksr [N]."""
        return self.a * self.ksf - self.b * self.ksr

    @property
    def damping_coupling(self) -> float:
        """Pitch-bounce damping coupling moment: a * csf - b * csr [N*s]."""
        return self.a * self.csf - self.b * self.csr


# Canonical default parameter singletons
DEFAULT_QUARTER_CAR_PARAMS = QuarterCarParameters()
DEFAULT_HALF_CAR_PARAMS = HalfCarParameters()
