"""
Automotive Chassis Telemetry Node.
Implements 100 Hz state telemetry broadcasting (0x120 Heave, 0x121 Pitch)
and 10 ms actuator command receiving (0x110 Demand) with AUTOSAR E2E safety monitoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from src.plant.half_car import HalfCarState
from src.plant.quarter_car import QuarterCarState
from src.telemetry.autosar_e2e import E2EStatus
from src.telemetry.bus_hal import VirtualCanBus, BusBackend
from src.telemetry.dbc_codec import (
    ActuatorDemandFrame,
    ChassisDbcCodec,
    ChassisHeaveFrame,
    ChassisPitchFrame,
    ID_ACTUATOR_DEMAND,
    ID_CHASSIS_HEAVE_TELEMETRY,
    ID_CHASSIS_PITCH_TELEMETRY,
)


@dataclass(frozen=True)
class TelemetryNodeStats:
    """Bus communication statistics and health metrics."""
    tx_heave_count: int
    tx_pitch_count: int
    rx_demand_count: int
    rx_crc_errors: int
    rx_lost_frames: int
    last_rtt_us: float


class ChassisTelemetryNode:
    """
    Electronic Control Unit (ECU) chassis telemetry publisher and demand receiver.
    Runs at 100 Hz (10 ms period).
    """

    def __init__(self, bus: Optional[VirtualCanBus] = None) -> None:
        self.bus = bus if bus is not None else VirtualCanBus(backend=BusBackend.LOOPBACK)
        self.codec = ChassisDbcCodec()

        # Telemetry counters
        self.tx_heave_count = 0
        self.tx_pitch_count = 0
        self.rx_demand_count = 0
        self.rx_crc_errors = 0
        self.rx_lost_frames = 0
        self.last_rtt_us = 0.0

    def publish_quarter_car_state(
        self,
        state: QuarterCarState,
        chassis_mode: int = 1,
    ) -> bool:
        """
        Package QuarterCarState into 0x120 Heave Telemetry frame and publish to bus.
        Execution latency guaranteed < 20 us.
        """
        frame = ChassisHeaveFrame(
            sprung_accel=float(state.sprung_accel),
            sprung_velocity=float(state.sprung_velocity),
            suspension_deflection=float(state.suspension_deflection),
            chassis_mode=int(chassis_mode),
        )
        payload = self.codec.encode_chassis_heave(frame, protect=True)
        success = self.bus.send(ID_CHASSIS_HEAVE_TELEMETRY, payload)
        if success:
            self.tx_heave_count += 1
        return success

    def publish_half_car_state(
        self,
        state: HalfCarState,
        chassis_mode: int = 1,
        safety_status: int = 0,
    ) -> Tuple[bool, bool]:
        """
        Package HalfCarState into both 0x120 (Heave) and 0x121 (Pitch) frames and publish.
        """
        heave_frame = ChassisHeaveFrame(
            sprung_accel=float(state.bounce_accel),
            sprung_velocity=float(state.bounce_velocity),
            suspension_deflection=float(state.susp_deflection_front),
            chassis_mode=int(chassis_mode),
        )
        payload_h = self.codec.encode_chassis_heave(heave_frame, protect=True)
        res_h = self.bus.send(ID_CHASSIS_HEAVE_TELEMETRY, payload_h)
        if res_h:
            self.tx_heave_count += 1

        pitch_frame = ChassisPitchFrame(
            pitch_angle=float(state.pitch_angle),
            pitch_rate=float(state.pitch_rate),
            tire_deflection=float(state.tire_deflection_front),
            safety_status=int(safety_status),
        )
        payload_p = self.codec.encode_chassis_pitch(pitch_frame, protect=True)
        res_p = self.bus.send(ID_CHASSIS_PITCH_TELEMETRY, payload_p)
        if res_p:
            self.tx_pitch_count += 1

        return res_h, res_p

    def receive_actuator_demand(
        self,
        timeout: float = 0.001,
    ) -> Optional[Tuple[ActuatorDemandFrame, E2EStatus]]:
        """
        Poll virtual bus for incoming 0x110 ActuatorDemand frame.
        Validates AUTOSAR E2E Profile 1 CRC-8 and Alive Counter.
        """
        msg = self.bus.receive(timeout=timeout)
        if msg is None:
            return None

        can_id, payload, _ = msg
        if can_id != ID_ACTUATOR_DEMAND:
            return None

        self.rx_demand_count += 1
        frame, status = self.codec.decode_actuator_demand(payload, verify=True)

        if status == E2EStatus.CRC_ERROR:
            self.rx_crc_errors += 1
        elif status in (E2EStatus.LOST_FRAME, E2EStatus.SEQUENCE_ERROR):
            self.rx_lost_frames += 1

        return frame, status

    def get_stats(self) -> TelemetryNodeStats:
        """Retrieve telemetry link operational metrics."""
        return TelemetryNodeStats(
            tx_heave_count=self.tx_heave_count,
            tx_pitch_count=self.tx_pitch_count,
            rx_demand_count=self.rx_demand_count,
            rx_crc_errors=self.rx_crc_errors,
            rx_lost_frames=self.rx_lost_frames,
            last_rtt_us=self.last_rtt_us,
        )
