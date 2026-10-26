"""
Physical Active Suspension Actuator Model and Diagnostics.

Enforces:
1. Physical force saturation bounds (+/- 2500 N).
2. Dynamic rate-of-change slew limiter (15,000 N/s = 150 N per 10 ms step).
3. 1st-order thermal dynamics model with heat generation and dissipation.
4. Automatic thermal derating (50% reduction when T > 110 deg C).
5. 16-bit diagnostic status word encoding for AUTOSAR / CAN-bus telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Optional, Tuple


class ActuatorStatusFlags(IntFlag):
    """Actuator diagnostic status bitfield flags (bits 0-7)."""
    NONE = 0
    SATURATION_UPPER = 1 << 0     # 0x0001: Clamped at positive physical force limit (+2500 N)
    SATURATION_LOWER = 1 << 1     # 0x0002: Clamped at negative physical force limit (-2500 N)
    SLEW_LIMIT_POSITIVE = 1 << 2  # 0x0004: Rate of force increase clamped by slew rate limit
    SLEW_LIMIT_NEGATIVE = 1 << 3  # 0x0008: Rate of force decrease clamped by slew rate limit
    ACTUATOR_ENABLED = 1 << 4     # 0x0010: Actuator power stage energized and operating
    THERMAL_WARNING = 1 << 5      # 0x0020: Coil/fluid temperature elevated (> 85 deg C)
    THERMAL_DERATING = 1 << 6     # 0x0040: Temperature critical (> 110 deg C), force derated 50%
    HARDWARE_FAULT = 1 << 7       # 0x0080: Hardware overcurrent or electrical fault


class ActuatorMode(IntEnum):
    """Actuator operating mode (bits 8-11)."""
    STANDBY = 0                   # Passive mode (F_a = 0)
    NORMAL_ACTIVE = 1             # Standard active suspension MPC + CBF
    COMFORT_PLUS = 2              # Active comfort mode with road preview lookahead
    SPORT = 3                     # Firm damping body roll/pitch suppression
    DEGRADED_SAFETY = 4           # Safety filter active only (MPC bypassed)
    CALIBRATION = 5               # Diagnostic actuator stroke test
    FAULT_SHUTDOWN = 15           # Emergency zero-force dump


class ActuatorHealthCode(IntEnum):
    """Actuator overall health code (bits 12-15)."""
    OK = 0                        # Normal nominal health
    WARNING = 1                   # Thermal warning or transient limiting
    DEGRADED = 2                  # Thermal derating or reduced capability
    CRITICAL = 3                  # Hardware fault or shutdown


@dataclass(frozen=True)
class ActuatorState:
    """Actuator operational state and diagnostic output."""
    force: float                  # Actual output force delivered to suspension [N]
    force_rate: float             # Instantaneous rate of force change [N/s]
    temperature: float            # Estimated coil/fluid temperature [deg C]
    flags: ActuatorStatusFlags    # Bitfield flags (lower 8 bits)
    mode: ActuatorMode            # Current operational mode
    health: ActuatorHealthCode    # Current health status code
    status_word: int              # Complete 16-bit status word
    energy_dissipated_j: float    # Cumulative mechanical energy dissipated [J]


class ActuatorModel:
    """
    Physical Active Suspension Strut Actuator Model.
    Simulates physical saturation, slew limiting, thermal dynamics, and AUTOSAR status.
    """

    def __init__(
        self,
        force_min: float = -2500.0,
        force_max: float = 2500.0,
        slew_rate_max: float = 15000.0,
        thermal_time_constant: float = 60.0,
        t_ambient: float = 25.0,
        t_warning: float = 85.0,
        t_critical: float = 110.0,
        thermal_resistance: float = 2.5e-6,
    ) -> None:
        self.force_min = float(force_min)
        self.force_max = float(force_max)
        self.slew_rate_max = float(slew_rate_max)
        self.tau_th = float(thermal_time_constant)
        self.t_amb = float(t_ambient)
        self.t_warn = float(t_warning)
        self.t_crit = float(t_critical)
        self.r_th = float(thermal_resistance)

        if self.force_min >= self.force_max:
            raise ValueError(f"force_min ({self.force_min}) must be < force_max ({self.force_max})")
        if self.slew_rate_max <= 0.0:
            raise ValueError(f"slew_rate_max ({self.slew_rate_max}) must be > 0")

        self.force_actual = 0.0
        self.temperature = self.t_amb
        self.enabled = True
        self.hardware_fault = False
        self.mode = ActuatorMode.NORMAL_ACTIVE
        self.total_energy_j = 0.0

    def reset(
        self,
        initial_force: float = 0.0,
        initial_temp: Optional[float] = None,
    ) -> None:
        """Reset internal actuator state."""
        self.force_actual = float(initial_force)
        self.temperature = float(initial_temp) if initial_temp is not None else self.t_amb
        self.hardware_fault = False
        self.total_energy_j = 0.0

    def set_mode(self, mode: ActuatorMode) -> None:
        """Change actuator operating mode."""
        self.mode = ActuatorMode(mode)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable actuator power stage."""
        self.enabled = bool(enabled)

    def trigger_fault(self, fault: bool = True) -> None:
        """Inject or clear hardware fault condition."""
        self.hardware_fault = bool(fault)

    def step(
        self,
        force_cmd: float,
        dt: float = 0.010,
        susp_vel: float = 0.0,
    ) -> ActuatorState:
        """
        Execute one discrete actuator update step of duration dt seconds.
        """
        dt = max(1e-6, float(dt))
        f_cmd = float(force_cmd)

        # 1. Check if disabled, faulted, or in standby
        if not self.enabled or self.mode == ActuatorMode.STANDBY or self.mode == ActuatorMode.FAULT_SHUTDOWN or self.hardware_fault:
            self.force_actual = 0.0
            flags = ActuatorStatusFlags.NONE
            if self.hardware_fault:
                flags |= ActuatorStatusFlags.HARDWARE_FAULT
            health = ActuatorHealthCode.CRITICAL if self.hardware_fault else ActuatorHealthCode.OK
            status_word = (int(flags) & 0xFF) | ((int(self.mode) & 0x0F) << 8) | ((int(health) & 0x0F) << 12)
            return ActuatorState(
                force=0.0,
                force_rate=0.0,
                temperature=self.temperature,
                flags=flags,
                mode=self.mode,
                health=health,
                status_word=status_word,
                energy_dissipated_j=self.total_energy_j,
            )

        flags = ActuatorStatusFlags.ACTUATOR_ENABLED

        # 2. Slew Rate Limiter
        max_delta = self.slew_rate_max * dt
        delta = f_cmd - self.force_actual

        if delta > max_delta:
            f_slew = self.force_actual + max_delta
            flags |= ActuatorStatusFlags.SLEW_LIMIT_POSITIVE
        elif delta < -max_delta:
            f_slew = self.force_actual - max_delta
            flags |= ActuatorStatusFlags.SLEW_LIMIT_NEGATIVE
        else:
            f_slew = f_cmd

        # 3. Thermal Dynamics and Derating
        # Thermal model: dT/dt = (T_amb - T) / tau_th + R_th * F^2
        if self.temperature > self.t_crit:
            derating = 0.5
            flags |= ActuatorStatusFlags.THERMAL_DERATING
        elif self.temperature > self.t_warn:
            derating = 1.0
            flags |= ActuatorStatusFlags.THERMAL_WARNING
        else:
            derating = 1.0

        eff_min = self.force_min * derating
        eff_max = self.force_max * derating

        # 4. Saturation Limiter
        if f_slew >= eff_max - 1e-4 or f_cmd >= eff_max - 1e-4:
            if f_slew >= eff_max - 1e-4:
                f_act = eff_max
            else:
                f_act = f_slew
            if f_cmd >= eff_max - 1e-4:
                flags |= ActuatorStatusFlags.SATURATION_UPPER
        elif f_slew <= eff_min + 1e-4 or f_cmd <= eff_min + 1e-4:
            if f_slew <= eff_min + 1e-4:
                f_act = eff_min
            else:
                f_act = f_slew
            if f_cmd <= eff_min + 1e-4:
                flags |= ActuatorStatusFlags.SATURATION_LOWER
        else:
            f_act = f_slew

        f_rate = (f_act - self.force_actual) / dt
        self.force_actual = f_act

        # Update thermal state
        d_temp = ((self.t_amb - self.temperature) / self.tau_th + self.r_th * (f_act ** 2)) * dt
        self.temperature += d_temp

        # Mechanical energy dissipated: power = |F * v|
        self.total_energy_j += abs(f_act * float(susp_vel)) * dt

        # Determine health code
        if flags & ActuatorStatusFlags.THERMAL_DERATING:
            health = ActuatorHealthCode.DEGRADED
        elif (flags & ActuatorStatusFlags.THERMAL_WARNING) or (flags & (ActuatorStatusFlags.SLEW_LIMIT_POSITIVE | ActuatorStatusFlags.SLEW_LIMIT_NEGATIVE)):
            health = ActuatorHealthCode.WARNING
        else:
            health = ActuatorHealthCode.OK

        # Encode 16-bit status word
        status_word = (
            (int(flags) & 0xFF)
            | ((int(self.mode) & 0x0F) << 8)
            | ((int(health) & 0x0F) << 12)
        )

        return ActuatorState(
            force=f_act,
            force_rate=f_rate,
            temperature=self.temperature,
            flags=flags,
            mode=self.mode,
            health=health,
            status_word=status_word,
            energy_dissipated_j=self.total_energy_j,
        )
