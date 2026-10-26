"""
Unit Tests for Vector DBC Codec Bitfield Serialization and Roundtrip Exactness.
"""

from __future__ import annotations

import math
import pytest

from src.telemetry.autosar_e2e import E2EStatus
from src.telemetry.dbc_codec import (
    ActuatorDemandFrame,
    ChassisDbcCodec,
    ChassisHeaveFrame,
    ChassisPitchFrame,
)


class TestDbcRoundtrip:
    """Test suite for DBC Codec serialization/deserialization precision."""

    def test_actuator_demand_roundtrip(self) -> None:
        """Verify ActuatorDemandFrame encodes and decodes with precision < 0.1 N."""
        codec = ChassisDbcCodec()

        test_cases = [
            (0.0, 0.0, 0, 0),
            (1250.5, -1250.5, 1, 0x01),
            (-2500.0, 2500.0, 2, 0x03),
            (350.2, -720.8, 3, 0x0F),
        ]

        for ff, fr, mode, flags in test_cases:
            orig = ActuatorDemandFrame(
                force_front=ff,
                force_rear=fr,
                control_mode=mode,
                actuator_status_flags=flags,
            )
            payload = codec.encode_actuator_demand(orig, protect=True)
            assert len(payload) == 8

            decoded, status = codec.decode_actuator_demand(payload, verify=True)
            assert status == E2EStatus.OK
            assert decoded.force_front == pytest.approx(ff, abs=0.1)
            assert decoded.force_rear == pytest.approx(fr, abs=0.1)
            assert decoded.control_mode == mode
            assert decoded.actuator_status_flags == flags

    def test_chassis_heave_roundtrip(self) -> None:
        """Verify ChassisHeaveFrame encodes and decodes within scale tolerance."""
        codec = ChassisDbcCodec()

        test_cases = [
            (0.0, 0.0, 0.0, 0),
            (1.234, -0.4567, 0.02540, 1),
            (-5.890, 1.2000, -0.04500, 2),
            (15.000, -2.5000, 0.05000, 3),
        ]

        for acc, vel, defl, mode in test_cases:
            orig = ChassisHeaveFrame(
                sprung_accel=acc,
                sprung_velocity=vel,
                suspension_deflection=defl,
                chassis_mode=mode,
            )
            payload = codec.encode_chassis_heave(orig, protect=True)
            assert len(payload) == 8

            decoded, status = codec.decode_chassis_heave(payload, verify=True)
            assert status == E2EStatus.OK
            assert decoded.sprung_accel == pytest.approx(acc, abs=0.001)
            assert decoded.sprung_velocity == pytest.approx(vel, abs=0.0001)
            assert decoded.suspension_deflection == pytest.approx(defl, abs=0.00001)
            assert decoded.chassis_mode == mode

    def test_chassis_pitch_roundtrip(self) -> None:
        """Verify ChassisPitchFrame encodes and decodes within scale tolerance."""
        codec = ChassisDbcCodec()

        test_cases = [
            (0.0, 0.0, 0.0, 0),
            (0.0123, -0.0456, 0.01500, 1),
            (-0.0500, 0.1200, -0.02500, 2),
        ]

        for angle, rate, tire_defl, safety in test_cases:
            orig = ChassisPitchFrame(
                pitch_angle=angle,
                pitch_rate=rate,
                tire_deflection=tire_defl,
                safety_status=safety,
            )
            payload = codec.encode_chassis_pitch(orig, protect=True)
            assert len(payload) == 8

            decoded, status = codec.decode_chassis_pitch(payload, verify=True)
            assert status == E2EStatus.OK
            assert decoded.pitch_angle == pytest.approx(angle, abs=0.0001)
            assert decoded.pitch_rate == pytest.approx(rate, abs=0.0001)
            assert decoded.tire_deflection == pytest.approx(tire_defl, abs=0.00001)
            assert decoded.safety_status == safety
