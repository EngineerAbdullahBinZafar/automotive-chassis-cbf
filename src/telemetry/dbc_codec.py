"""
Vector DBC Codec for Automotive Chassis Dynamics and Actuator Commands.
Implements high-performance pure-Python bitfield serialization and unpacking,
with optional Vector DBC parsing via cantools.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from src.telemetry.autosar_e2e import E2EProfile1, E2EStatus


# CAN Message IDs
ID_ACTUATOR_DEMAND = 0x110         # 272 decimal
ID_CHASSIS_HEAVE_TELEMETRY = 0x120  # 288 decimal
ID_CHASSIS_PITCH_TELEMETRY = 0x121  # 289 decimal


@dataclass(frozen=True)
class ActuatorDemandFrame:
    """CAN Frame 0x110: Actuator Force Command from Active Suspension Controller."""
    force_front: float          # [N], scale: 0.1, range: [-2500, 2500]
    force_rear: float           # [N], scale: 0.1, range: [-2500, 2500]
    control_mode: int           # [enum: 0=PASSIVE, 1=COMFORT_MPC, 2=SAFETY_CBF, 3=EMERGENCY]
    actuator_status_flags: int  # [bitmask: bit0=SATURATED, bit1=SLEW_LIMITED, bit2=DERATED, bit3=FAULT]
    alive_counter: int = 0      # [0..14]
    crc8: int = 0               # [0..255]


@dataclass(frozen=True)
class ChassisHeaveFrame:
    """CAN Frame 0x120: Chassis Heave Dynamics Telemetry (100 Hz)."""
    sprung_accel: float         # [m/s^2], scale: 0.001, range: [-30, 30]
    sprung_velocity: float      # [m/s], scale: 0.0001, range: [-3, 3]
    suspension_deflection: float# [m], scale: 0.00001 (10 um), range: [-0.1, 0.1]
    chassis_mode: int           # [0..15]
    alive_counter: int = 0      # [0..14]
    crc8: int = 0               # [0..255]


@dataclass(frozen=True)
class ChassisPitchFrame:
    """CAN Frame 0x121: Chassis Pitch & Ground Contact Telemetry (100 Hz)."""
    pitch_angle: float          # [rad], scale: 0.0001, range: [-1, 1]
    pitch_rate: float           # [rad/s], scale: 0.0001, range: [-2, 2]
    tire_deflection: float      # [m], scale: 0.00001 (10 um), range: [-0.1, 0.1]
    safety_status: int          # [0..15: 0=SAFE, 1=STROKE_WARN, 2=TIRE_WARN, 3=CBF_ACTIVE]
    alive_counter: int = 0      # [0..14]
    crc8: int = 0               # [0..255]


class ChassisDbcCodec:
    """
    High-performance encoder and decoder for chassis CAN frames.
    Encodes floating-point physical signals to fixed-point integers
    according to chassis_dynamics.dbc scaling and little-endian packing.
    """

    def __init__(self) -> None:
        self.e2e_demand = E2EProfile1(data_id=ID_ACTUATOR_DEMAND)
        self.e2e_heave = E2EProfile1(data_id=ID_CHASSIS_HEAVE_TELEMETRY)
        self.e2e_pitch = E2EProfile1(data_id=ID_CHASSIS_PITCH_TELEMETRY)

    def encode_actuator_demand(
        self,
        frame: ActuatorDemandFrame,
        protect: bool = True,
    ) -> bytes:
        """Encode ActuatorDemandFrame to 8-byte CAN payload."""
        # Scale: 0.1 N per count -> raw = round(force / 0.1) = round(force * 10)
        raw_front = max(-32768, min(32767, int(round(frame.force_front * 10.0))))
        raw_rear = max(-32768, min(32767, int(round(frame.force_rear * 10.0))))
        raw_mode = max(0, min(255, int(frame.control_mode)))
        raw_flags = max(0, min(255, int(frame.actuator_status_flags)))

        buf = bytearray(8)
        struct.pack_into("<hhBB", buf, 0, raw_front, raw_rear, raw_mode, raw_flags)

        if protect:
            self.e2e_demand.protect(buf)
        else:
            buf[6] = frame.alive_counter & 0x0F
            buf[7] = frame.crc8 & 0xFF

        return bytes(buf)

    def decode_actuator_demand(
        self,
        payload: bytes,
        verify: bool = True,
    ) -> Tuple[ActuatorDemandFrame, E2EStatus]:
        """Decode 8-byte CAN payload into ActuatorDemandFrame."""
        if len(payload) != 8:
            raise ValueError(f"Expected 8 bytes for ActuatorDemand, got {len(payload)}")

        status = self.e2e_demand.verify(payload)[0] if verify else E2EStatus.OK

        raw_front, raw_rear, raw_mode, raw_flags = struct.unpack_from("<hhBB", payload, 0)
        counter = payload[6] & 0x0F
        crc = payload[7]

        frame = ActuatorDemandFrame(
            force_front=float(raw_front * 0.1),
            force_rear=float(raw_rear * 0.1),
            control_mode=raw_mode,
            actuator_status_flags=raw_flags,
            alive_counter=counter,
            crc8=crc,
        )
        return frame, status

    def encode_chassis_heave(
        self,
        frame: ChassisHeaveFrame,
        protect: bool = True,
    ) -> bytes:
        """Encode ChassisHeaveFrame to 8-byte CAN payload."""
        # Sprung Accel: 0.001 m/s^2 per count -> raw = round(accel * 1000.0)
        raw_acc = max(-32768, min(32767, int(round(frame.sprung_accel * 1000.0))))
        # Sprung Velocity: 0.0001 m/s per count -> raw = round(vel * 10000.0)
        raw_vel = max(-32768, min(32767, int(round(frame.sprung_velocity * 10000.0))))
        # Suspension Deflection: 0.00001 m per count -> raw = round(defl * 100000.0)
        raw_defl = max(-32768, min(32767, int(round(frame.suspension_deflection * 100000.0))))

        buf = bytearray(8)
        struct.pack_into("<hhh", buf, 0, raw_acc, raw_vel, raw_defl)
        buf[6] = (frame.chassis_mode & 0x0F) << 4

        if protect:
            self.e2e_heave.protect(buf)
        else:
            buf[6] |= frame.alive_counter & 0x0F
            buf[7] = frame.crc8 & 0xFF

        return bytes(buf)

    def decode_chassis_heave(
        self,
        payload: bytes,
        verify: bool = True,
    ) -> Tuple[ChassisHeaveFrame, E2EStatus]:
        """Decode 8-byte CAN payload into ChassisHeaveFrame."""
        if len(payload) != 8:
            raise ValueError(f"Expected 8 bytes for ChassisHeave, got {len(payload)}")

        status = self.e2e_heave.verify(payload)[0] if verify else E2EStatus.OK

        raw_acc, raw_vel, raw_defl = struct.unpack_from("<hhh", payload, 0)
        counter = payload[6] & 0x0F
        mode = (payload[6] >> 4) & 0x0F
        crc = payload[7]

        frame = ChassisHeaveFrame(
            sprung_accel=float(raw_acc * 0.001),
            sprung_velocity=float(raw_vel * 0.0001),
            suspension_deflection=float(raw_defl * 0.00001),
            chassis_mode=mode,
            alive_counter=counter,
            crc8=crc,
        )
        return frame, status

    def encode_chassis_pitch(
        self,
        frame: ChassisPitchFrame,
        protect: bool = True,
    ) -> bytes:
        """Encode ChassisPitchFrame to 8-byte CAN payload."""
        # Pitch Angle: 0.0001 rad per count
        raw_angle = max(-32768, min(32767, int(round(frame.pitch_angle * 10000.0))))
        # Pitch Rate: 0.0001 rad/s per count
        raw_rate = max(-32768, min(32767, int(round(frame.pitch_rate * 10000.0))))
        # Tire Deflection: 0.00001 m per count
        raw_tire = max(-32768, min(32767, int(round(frame.tire_deflection * 100000.0))))

        buf = bytearray(8)
        struct.pack_into("<hhh", buf, 0, raw_angle, raw_rate, raw_tire)
        buf[6] = (frame.safety_status & 0x0F) << 4

        if protect:
            self.e2e_pitch.protect(buf)
        else:
            buf[6] |= frame.alive_counter & 0x0F
            buf[7] = frame.crc8 & 0xFF

        return bytes(buf)

    def decode_chassis_pitch(
        self,
        payload: bytes,
        verify: bool = True,
    ) -> Tuple[ChassisPitchFrame, E2EStatus]:
        """Decode 8-byte CAN payload into ChassisPitchFrame."""
        if len(payload) != 8:
            raise ValueError(f"Expected 8 bytes for ChassisPitch, got {len(payload)}")

        status = self.e2e_pitch.verify(payload)[0] if verify else E2EStatus.OK

        raw_angle, raw_rate, raw_tire = struct.unpack_from("<hhh", payload, 0)
        counter = payload[6] & 0x0F
        safety = (payload[6] >> 4) & 0x0F
        crc = payload[7]

        frame = ChassisPitchFrame(
            pitch_angle=float(raw_angle * 0.0001),
            pitch_rate=float(raw_rate * 0.0001),
            tire_deflection=float(raw_tire * 0.00001),
            safety_status=safety,
            alive_counter=counter,
            crc8=crc,
        )
        return frame, status
