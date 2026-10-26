"""
Automotive Telemetry and CAN Bus Communication Module (Milestone 3).
"""

from __future__ import annotations

from src.telemetry.autosar_e2e import (
    E2EProfile1,
    E2EStatus,
    calculate_crc8_sae_j1850,
)
from src.telemetry.bus_hal import (
    BusBackend,
    VirtualCanBus,
)
from src.telemetry.dbc_codec import (
    ActuatorDemandFrame,
    ChassisDbcCodec,
    ChassisHeaveFrame,
    ChassisPitchFrame,
    ID_ACTUATOR_DEMAND,
    ID_CHASSIS_HEAVE_TELEMETRY,
    ID_CHASSIS_PITCH_TELEMETRY,
)
from src.telemetry.telemetry_node import (
    ChassisTelemetryNode,
    TelemetryNodeStats,
)

__all__ = [
    "E2EProfile1",
    "E2EStatus",
    "calculate_crc8_sae_j1850",
    "BusBackend",
    "VirtualCanBus",
    "ActuatorDemandFrame",
    "ChassisDbcCodec",
    "ChassisHeaveFrame",
    "ChassisPitchFrame",
    "ID_ACTUATOR_DEMAND",
    "ID_CHASSIS_HEAVE_TELEMETRY",
    "ID_CHASSIS_PITCH_TELEMETRY",
    "ChassisTelemetryNode",
    "TelemetryNodeStats",
]
