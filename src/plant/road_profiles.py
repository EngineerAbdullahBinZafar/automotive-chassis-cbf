"""
ISO 8608 Road Roughness Profile Generator and Discrete Obstacles.

Features:
- ISO 8608 Road Classes A through E Ornstein-Uhlenbeck stochastic shaping filter.
- Exact sample-invariant discrete-time formulation (zero variance distortion across dt).
- Circular FIFO buffer with sub-sample fractional interpolation for rear wheelbase delay.
- Discrete obstacle profiles: Half-sine speed bump (50 mm) and negative pothole (-40 mm).
- Combined multi-axle road excitation generation.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class RoadClass(enum.Enum):
    """ISO 8608 Road Roughness Classification (displacement PSD Gd(n0) in m^3)."""
    CLASS_A = "A"  # Motorway / Autobahn: 32e-6 m^3
    CLASS_B = "B"  # Major Road / Good Asphalt: 64e-6 m^3
    CLASS_C = "C"  # Average Road / Rural Secondary: 256e-6 m^3
    CLASS_D = "D"  # Poor Pavement / Rough: 1024e-6 m^3
    CLASS_E = "E"  # Very Poor / Cobblestone: 4096e-6 m^3


# ISO 8608 Reference Constants
ISO_GD_N0_MAP: dict[Union[RoadClass, str], float] = {
    RoadClass.CLASS_A: 32e-6,
    RoadClass.CLASS_B: 64e-6,
    RoadClass.CLASS_C: 256e-6,
    RoadClass.CLASS_D: 1024e-6,
    RoadClass.CLASS_E: 4096e-6,
    "A": 32e-6,
    "B": 64e-6,
    "C": 256e-6,
    "D": 1024e-6,
    "E": 4096e-6,
}
REFERENCE_SPATIAL_FREQ: float = 0.1       # n0 = 0.1 cycles/m
DEFAULT_CUTOFF_SPATIAL_FREQ: float = 0.05 # f0 = 0.05 cycles/m


@dataclass(frozen=True)
class RoadSurfacePoint:
    """Road excitation state at a single contact patch."""
    elevation: float  # z_r in meters
    velocity: float   # dz_r/dt in m/s


@dataclass(frozen=True)
class DualAxleRoadPoint:
    """Synchronized front and delayed rear road excitations for half-car plant."""
    front: RoadSurfacePoint
    rear: RoadSurfacePoint


class WheelbaseDelayBuffer:
    """
    Circular FIFO ring buffer providing sub-sample linearly interpolated
    wheelbase delay tau = L / v with O(1) complexity and zero dynamic allocations.
    """

    def __init__(
        self,
        wheelbase: float = 3.20,
        dt: float = 0.001,
        min_velocity: float = 1.0,
    ) -> None:
        self.wheelbase = float(wheelbase)
        self.dt = float(dt)
        self.min_velocity = max(0.1, float(min_velocity))
        # Max delay steps possible at min_velocity plus safety buffer
        self.capacity = int(math.ceil(self.wheelbase / (self.min_velocity * self.dt))) + 32
        self.elevations = [0.0] * self.capacity
        self.velocities = [0.0] * self.capacity
        self.head = 0
        self.count = 0

    def push(self, point: RoadSurfacePoint) -> None:
        """Push latest front wheel road point into circular buffer."""
        self.elevations[self.head] = point.elevation
        self.velocities[self.head] = point.velocity
        self.head = (self.head + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def get_delayed(self, velocity: float) -> RoadSurfacePoint:
        """Retrieve delayed road point for rear wheel at given vehicle forward velocity."""
        v = max(self.min_velocity, abs(float(velocity)))
        tau = self.wheelbase / v
        delay_steps = tau / self.dt

        n = int(math.floor(delay_steps))
        delta = delay_steps - n

        # head - 1 is current sample (delay = 0)
        idx0 = (self.head - 1 - n) % self.capacity
        idx1 = (self.head - 1 - n - 1) % self.capacity

        z_del = (1.0 - delta) * self.elevations[idx0] + delta * self.elevations[idx1]
        v_del = (1.0 - delta) * self.velocities[idx0] + delta * self.velocities[idx1]
        return RoadSurfacePoint(elevation=z_del, velocity=v_del)

    def reset(self) -> None:
        """Clear buffer history."""
        self.elevations = [0.0] * self.capacity
        self.velocities = [0.0] * self.capacity
        self.head = 0
        self.count = 0


class ISO8608RoadGenerator:
    """
    Exact sample-invariant Ornstein-Uhlenbeck discrete road roughness generator.

    Formula:
        z_r[k+1] = Phi * z_r[k] + Gamma * xi[k]
        Phi = exp(-2*pi*f0*v*dt)
        V_theo = (pi * n0^2 * Gd(n0)) / (2 * f0)
        Gamma = sqrt(V_theo * (1 - Phi^2))
    """

    def __init__(
        self,
        road_class: Union[RoadClass, str] = RoadClass.CLASS_C,
        velocity: float = 20.0,
        dt: float = 0.001,
        seed: Optional[int] = 42,
        f0: float = DEFAULT_CUTOFF_SPATIAL_FREQ,
        n0: float = REFERENCE_SPATIAL_FREQ,
    ) -> None:
        if isinstance(road_class, RoadClass):
            self.road_class = road_class
        else:
            self.road_class = RoadClass(road_class.upper())
        self.gd_n0 = ISO_GD_N0_MAP[self.road_class]
        self.velocity = float(velocity)
        self.dt = float(dt)
        self.f0 = float(f0)
        self.n0 = float(n0)
        self.seed = seed

        if HAS_NUMPY:
            self._rng = np.random.default_rng(seed)
        else:
            import random
            self._rng = random.Random(seed)

        self._recompute_coefficients()
        self.z_current = 0.0

    def _recompute_coefficients(self) -> None:
        self.alpha = 2.0 * math.pi * self.f0 * self.velocity
        self.phi = math.exp(-self.alpha * self.dt)
        self.var_theo = (math.pi * (self.n0 ** 2) * self.gd_n0) / (2.0 * self.f0)
        self.gamma = math.sqrt(self.var_theo * (1.0 - self.phi ** 2))

    def set_velocity(self, velocity: float) -> None:
        """Update vehicle forward velocity and recompute shaping filter coefficients."""
        self.velocity = max(0.1, float(velocity))
        self._recompute_coefficients()

    def step(self) -> RoadSurfacePoint:
        """Advance road profile by one discrete time step dt."""
        if HAS_NUMPY:
            xi = float(self._rng.standard_normal())
        else:
            # Box-Muller transform for pure-Python Gaussian standard normal
            u1 = max(1e-15, self._rng.random())
            u2 = self._rng.random()
            xi = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

        z_next = self.phi * self.z_current + self.gamma * xi
        v_current = (z_next - self.z_current) / self.dt
        self.z_current = z_next
        return RoadSurfacePoint(elevation=z_next, velocity=v_current)

    def generate_profile(self, duration: float) -> Tuple[List[float], List[float], List[float]]:
        """Pre-compute a trajectory over duration [seconds]. Returns (t, z_r, v_r)."""
        steps = int(math.ceil(duration / self.dt))
        t_list = [i * self.dt for i in range(steps)]
        z_list = [0.0] * steps
        v_list = [0.0] * steps
        for i in range(steps):
            pt = self.step()
            z_list[i] = pt.elevation
            v_list[i] = pt.velocity
        return t_list, z_list, v_list

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal filter state and optionally re-seed RNG."""
        self.z_current = 0.0
        if seed is not None:
            self.seed = seed
            if HAS_NUMPY:
                self._rng = np.random.default_rng(seed)
            else:
                self._rng.seed(seed)


class SpeedBump:
    """Half-sine speed bump profile (H = 50 mm, L = 0.5 m)."""

    def __init__(
        self,
        height: float = 0.050,
        length: float = 0.50,
        start_position: float = 5.0,
    ) -> None:
        self.height = float(height)
        self.length = float(length)
        self.start_position = float(start_position)
        self.end_position = self.start_position + self.length

    def evaluate(self, x: float, v: float) -> RoadSurfacePoint:
        """Evaluate bump elevation and vertical velocity at vehicle position x and speed v."""
        if self.start_position <= x <= self.end_position:
            arg = (math.pi * (x - self.start_position)) / self.length
            z = self.height * math.sin(arg)
            dz_dx = (math.pi * self.height / self.length) * math.cos(arg)
            return RoadSurfacePoint(elevation=z, velocity=dz_dx * v)
        return RoadSurfacePoint(elevation=0.0, velocity=0.0)


class Pothole:
    """Negative half-sine pothole profile (depth = 40 mm, L = 0.4 m)."""

    def __init__(
        self,
        depth: float = 0.040,
        length: float = 0.40,
        start_position: float = 5.0,
    ) -> None:
        self.depth = float(depth)
        self.length = float(length)
        self.start_position = float(start_position)
        self.end_position = self.start_position + self.length

    def evaluate(self, x: float, v: float) -> RoadSurfacePoint:
        """Evaluate pothole elevation and vertical velocity at vehicle position x and speed v."""
        if self.start_position <= x <= self.end_position:
            arg = (math.pi * (x - self.start_position)) / self.length
            z = -self.depth * math.sin(arg)
            dz_dx = -(math.pi * self.depth / self.length) * math.cos(arg)
            return RoadSurfacePoint(elevation=z, velocity=dz_dx * v)
        return RoadSurfacePoint(elevation=0.0, velocity=0.0)


class CombinedRoadProfile:
    """
    Unified multi-axle road excitation combining ISO 8608 roughness,
    discrete obstacles, and sub-sample wheelbase delay.
    """

    def __init__(
        self,
        road_class: Optional[Union[RoadClass, str]] = RoadClass.CLASS_C,
        obstacles: Optional[List[Union[SpeedBump, Pothole]]] = None,
        velocity: float = 20.0,
        dt: float = 0.001,
        wheelbase: float = 3.20,
        seed: Optional[int] = 42,
    ) -> None:
        self.velocity = float(velocity)
        self.dt = float(dt)
        self.wheelbase = float(wheelbase)
        self.x = 0.0  # Vehicle longitudinal position
        self.obstacles = obstacles if obstacles is not None else []

        self.iso_gen: Optional[ISO8608RoadGenerator] = None
        if road_class is not None:
            self.iso_gen = ISO8608RoadGenerator(
                road_class=road_class,
                velocity=self.velocity,
                dt=self.dt,
                seed=seed,
            )
        self.delay_buffer = WheelbaseDelayBuffer(
            wheelbase=self.wheelbase,
            dt=self.dt,
            min_velocity=1.0,
        )

    def step(self) -> DualAxleRoadPoint:
        """Advance simulation by dt, returning synchronized front and rear wheel points."""
        z_iso = 0.0
        v_iso = 0.0
        if self.iso_gen is not None:
            pt_iso = self.iso_gen.step()
            z_iso = pt_iso.elevation
            v_iso = pt_iso.velocity

        z_obs = 0.0
        v_obs = 0.0
        for obs in self.obstacles:
            pt_obs = obs.evaluate(self.x, self.velocity)
            z_obs += pt_obs.elevation
            v_obs += pt_obs.velocity

        front_pt = RoadSurfacePoint(elevation=z_iso + z_obs, velocity=v_iso + v_obs)
        self.delay_buffer.push(front_pt)
        rear_pt = self.delay_buffer.get_delayed(self.velocity)

        self.x += self.velocity * self.dt
        return DualAxleRoadPoint(front=front_pt, rear=rear_pt)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset road position and internal generators."""
        self.x = 0.0
        if self.iso_gen is not None:
            self.iso_gen.reset(seed)
        self.delay_buffer.reset()
