"""
Integration Tests for Virtual CAN Bus, Cross-Platform Communication, and Telemetry Node.
"""

from __future__ import annotations

import time
import pytest

from src.plant.quarter_car import QuarterCarState
from src.plant.half_car import HalfCarState
from src.telemetry.autosar_e2e import E2EStatus
from src.telemetry.bus_hal import VirtualCanBus, BusBackend
from src.telemetry.dbc_codec import (
    ActuatorDemandFrame,
    ID_ACTUATOR_DEMAND,
    ID_CHASSIS_HEAVE_TELEMETRY,
    ID_CHASSIS_PITCH_TELEMETRY,
)
from src.telemetry.telemetry_node import ChassisTelemetryNode


class TestVirtualBusIntegration:
    """Test suite for virtual bus communication and end-to-end telemetry loop."""

    def test_loopback_bus_basic_send_receive(self) -> None:
        """Verify virtual bus sends and receives 8-byte frames synchronously."""
        bus = VirtualCanBus(backend=BusBackend.LOOPBACK)
        payload = bytes([1, 2, 3, 4, 5, 6, 7, 8])

        success = bus.send(ID_ACTUATOR_DEMAND, payload)
        assert success is True

        received = bus.receive(timeout=0.010)
        assert received is not None
        can_id, rx_data, ts = received
        assert can_id == ID_ACTUATOR_DEMAND
        assert rx_data == payload
        assert ts > 0.0
        bus.close()

    def test_virtual_bus_latency_distribution(self) -> None:
        """Verify in-memory loopback latency is strictly < 50 microseconds."""
        bus = VirtualCanBus(backend=BusBackend.LOOPBACK)
        payload = bytes([10, 20, 30, 40, 50, 60, 70, 80])

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            bus.send(0x120, payload)
            msg = bus.receive(timeout=0.001)
            t1 = time.perf_counter()
            assert msg is not None
            latencies.append((t1 - t0) * 1e6)

        median_latency = sorted(latencies)[len(latencies) // 2]
        print(f"\nVirtual Bus Median Latency: {median_latency:.2f} us")
        assert median_latency < 50.0  # Must be sub-50 microseconds
        bus.close()

    def test_telemetry_node_quarter_car_publishing(self) -> None:
        """Verify TelemetryNode encodes and publishes quarter car state."""
        bus = VirtualCanBus(backend=BusBackend.LOOPBACK)
        node = ChassisTelemetryNode(bus=bus)

        st = QuarterCarState(
            suspension_deflection=0.015,
            sprung_velocity=-0.25,
            tire_deflection=-0.005,
            unsprung_velocity=0.10,
            sprung_accel=1.85,
        )

        success = node.publish_quarter_car_state(st, chassis_mode=2)
        assert success is True
        assert node.tx_heave_count == 1

        msg = bus.receive(timeout=0.010)
        assert msg is not None
        can_id, payload, _ = msg
        assert can_id == ID_CHASSIS_HEAVE_TELEMETRY

        frame, status = node.codec.decode_chassis_heave(payload, verify=True)
        assert status == E2EStatus.OK
        assert frame.sprung_accel == pytest.approx(1.85, abs=0.002)
        assert frame.sprung_velocity == pytest.approx(-0.25, abs=0.001)
        assert frame.suspension_deflection == pytest.approx(0.015, abs=0.0001)
        assert frame.chassis_mode == 2
        bus.close()

    def test_telemetry_node_actuator_demand_reception(self) -> None:
        """Verify TelemetryNode receives and decodes actuator commands."""
        bus = VirtualCanBus(backend=BusBackend.LOOPBACK)
        node = ChassisTelemetryNode(bus=bus)

        # Simulate controller ECU sending demand
        demand = ActuatorDemandFrame(
            force_front=1500.0,
            force_rear=-800.0,
            control_mode=1,
            actuator_status_flags=0x01,
        )
        payload = node.codec.encode_actuator_demand(demand, protect=True)
        bus.send(ID_ACTUATOR_DEMAND, payload)

        rx_frame, status = node.receive_actuator_demand(timeout=0.010)
        assert rx_frame is not None
        assert status == E2EStatus.OK
        assert rx_frame.force_front == pytest.approx(1500.0, abs=0.2)
        assert rx_frame.force_rear == pytest.approx(-800.0, abs=0.2)
        assert rx_frame.control_mode == 1
        assert rx_frame.actuator_status_flags == 0x01
        assert node.rx_demand_count == 1
        assert node.rx_crc_errors == 0
        bus.close()
