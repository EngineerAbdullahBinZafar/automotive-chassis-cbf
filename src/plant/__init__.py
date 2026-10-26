"""
Vehicle dynamics plant models and road excitation subsystem.
Part of Milestone 1 for Automotive Active Suspension & CAN-Bus Dynamics Controller.
"""

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
from src.plant.road_profiles import (
    RoadClass,
    ISO_GD_N0_MAP,
    REFERENCE_SPATIAL_FREQ,
    DEFAULT_CUTOFF_SPATIAL_FREQ,
    RoadSurfacePoint,
    DualAxleRoadPoint,
    WheelbaseDelayBuffer,
    ISO8608RoadGenerator,
    SpeedBump,
    Pothole,
    CombinedRoadProfile,
)
from src.plant.integrator import (
    rk4_step_scalar,
    rk4_step_numpy,
    rk4_step_linear,
    FixedStepRK4Integrator,
    verify_linear_rk4_stability,
)

__all__ = [
    # Parameters
    "QuarterCarParameters",
    "DEFAULT_QUARTER_CAR_PARAMS",
    "HalfCarParameters",
    "DEFAULT_HALF_CAR_PARAMS",
    # Quarter-car
    "QuarterCarModel",
    "QuarterCarState",
    "ConstraintStatus",
    # Half-car
    "HalfCarModel",
    "HalfCarState",
    # Road profiles
    "RoadClass",
    "ISO_GD_N0_MAP",
    "REFERENCE_SPATIAL_FREQ",
    "DEFAULT_CUTOFF_SPATIAL_FREQ",
    "RoadSurfacePoint",
    "DualAxleRoadPoint",
    "WheelbaseDelayBuffer",
    "ISO8608RoadGenerator",
    "SpeedBump",
    "Pothole",
    "CombinedRoadProfile",
    # Integrators
    "rk4_step_scalar",
    "rk4_step_numpy",
    "rk4_step_linear",
    "FixedStepRK4Integrator",
    "verify_linear_rk4_stability",
]
