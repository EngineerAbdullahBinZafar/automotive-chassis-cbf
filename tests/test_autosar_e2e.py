"""
Unit Tests for AUTOSAR E2E Profile 1 (SAE J1850 CRC-8 and Alive Counter).
"""

from __future__ import annotations

import pytest
from src.telemetry.autosar_e2e import (
    E2EProfile1,
    E2EStatus,
    calculate_crc8_sae_j1850,
)


class TestAutosarE2E:
    """Test suite for AUTOSAR E2E Profile 1."""

    def test_crc8_sae_j1850_known_vector(self) -> None:
        """Verify CRC calculation against known test vectors."""
        data = bytes([0x00, 0x00, 0x00, 0x00])
        crc = calculate_crc8_sae_j1850(data, data_id=0x0000)
        assert isinstance(crc, int)
        assert 0 <= crc <= 255

        # Inverting single bit must change CRC
        data_corrupt = bytes([0x01, 0x00, 0x00, 0x00])
        crc_corrupt = calculate_crc8_sae_j1850(data_corrupt, data_id=0x0000)
        assert crc != crc_corrupt

    def test_data_id_authentication(self) -> None:
        """Verify changing Data ID alters CRC even for identical payload."""
        data = bytes([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x00])
        crc1 = calculate_crc8_sae_j1850(data, data_id=0x0110)
        crc2 = calculate_crc8_sae_j1850(data, data_id=0x0120)
        assert crc1 != crc2

    def test_alive_counter_rolling_sequence(self) -> None:
        """Verify alive counter rolls strictly from 0 to 14, then resets to 0."""
        e2e = E2EProfile1(data_id=0x0110)
        counters = []
        for _ in range(30):
            buf = bytearray(8)
            e2e.protect(buf)
            counters.append(buf[6] & 0x0F)

        expected_sequence = list(range(15)) + list(range(15))
        assert counters == expected_sequence

    def test_e2e_protect_and_verify_success(self) -> None:
        """Verify nominal sequence of frames passes verification."""
        tx = E2EProfile1(data_id=0x0120)
        rx = E2EProfile1(data_id=0x0120)

        for _ in range(20):
            buf = bytearray([1, 2, 3, 4, 5, 6, 0, 0])
            tx.protect(buf)
            status, counter = rx.verify(bytes(buf))
            assert status == E2EStatus.OK

    def test_crc_corruption_detection(self) -> None:
        """Verify bit corruption in payload triggers CRC_ERROR."""
        tx = E2EProfile1(data_id=0x0120)
        rx = E2EProfile1(data_id=0x0120)

        buf = bytearray([10, 20, 30, 40, 50, 60, 0, 0])
        tx.protect(buf)

        # Corrupt one payload byte
        buf[2] ^= 0x01
        status, _ = rx.verify(bytes(buf))
        assert status == E2EStatus.CRC_ERROR

    def test_repeated_message_detection(self) -> None:
        """Verify receiving identical counter twice triggers REPEATED status."""
        tx = E2EProfile1(data_id=0x0110)
        rx = E2EProfile1(data_id=0x0110)

        buf1 = bytearray([1, 1, 1, 1, 1, 1, 0, 0])
        tx.protect(buf1)
        status1, _ = rx.verify(bytes(buf1))
        assert status1 == E2EStatus.OK

        # Send same buffer again without advancing counter
        status2, _ = rx.verify(bytes(buf1))
        assert status2 == E2EStatus.REPEATED

    def test_lost_frame_detection(self) -> None:
        """Verify gap of 2 or 3 in alive counter triggers LOST_FRAME status."""
        tx = E2EProfile1(data_id=0x0110, max_allowed_delta=3)
        rx = E2EProfile1(data_id=0x0110, max_allowed_delta=3)

        buf0 = bytearray(8)
        tx.protect(buf0)
        status0, _ = rx.verify(bytes(buf0))
        assert status0 == E2EStatus.OK

        # Drop 1 frame in transmission
        buf_drop = bytearray(8)
        tx.protect(buf_drop)  # counter = 1 (dropped)

        buf2 = bytearray(8)
        tx.protect(buf2)      # counter = 2
        status2, counter2 = rx.verify(bytes(buf2))
        assert status2 == E2EStatus.LOST_FRAME
        assert counter2 == 2
