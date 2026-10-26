"""
AUTOSAR End-to-End (E2E) Communication Protection Profile 1.
Implements SAE J1850 CRC-8 verification, 4-bit rolling alive counter,
and Data ID authentication for safety-critical automotive chassis CAN frames.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Tuple


class E2EStatus(enum.Enum):
    """AUTOSAR E2E check result status."""
    OK = "OK"
    CRC_ERROR = "CRC_ERROR"
    REPEATED = "REPEATED"
    LOST_FRAME = "LOST_FRAME"
    SEQUENCE_ERROR = "SEQUENCE_ERROR"
    INVALID_LENGTH = "INVALID_LENGTH"


def calculate_crc8_sae_j1850(data: bytes, data_id: int = 0) -> int:
    """
    Calculate 8-bit CRC according to AUTOSAR E2E Profile 1 (SAE J1850).
    Generator polynomial: 0x1D (x^8 + x^4 + x^3 + x^2 + 1)
    Initial value: 0xFF
    Final XOR: 0xFF
    Data ID: 16-bit integer included as low-byte then high-byte.
    """
    crc = 0xFF
    poly = 0x1D

    # Include Data ID low and high bytes
    prefix = bytes([data_id & 0xFF, (data_id >> 8) & 0xFF])
    payload = prefix + data

    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc ^ 0xFF


class E2EProfile1:
    """
    AUTOSAR E2E Profile 1 protector and validator.
    Manages 4-bit alive counter [0..14] rolling and CRC-8 calculation.
    """

    def __init__(self, data_id: int, max_allowed_delta: int = 3) -> None:
        self.data_id = int(data_id)
        self.max_allowed_delta = int(max_allowed_delta)
        self._tx_counter = 0
        self._rx_last_counter = -1
        self._rx_first_message = True

    def reset(self) -> None:
        """Reset transmitter and receiver sequence counters."""
        self._tx_counter = 0
        self._rx_last_counter = -1
        self._rx_first_message = True

    def protect(self, payload: bytearray) -> bytearray:
        """
        Apply E2E protection to an 8-byte CAN payload.
        Byte 6 low-nibble receives the rolling alive counter [0..14].
        Byte 7 receives the SAE J1850 CRC-8 calculated over bytes 0..6.
        """
        if len(payload) != 8:
            raise ValueError(f"E2E Profile 1 requires exactly 8 bytes, got {len(payload)}")

        # Insert alive counter into byte 6 low nibble (preserving high nibble)
        counter = self._tx_counter
        payload[6] = (payload[6] & 0xF0) | (counter & 0x0F)

        # Advance rolling counter (0..14 rollover)
        self._tx_counter = (self._tx_counter + 1) % 15

        # Compute CRC8 over bytes 0..6
        crc = calculate_crc8_sae_j1850(bytes(payload[:7]), self.data_id)
        payload[7] = crc

        return payload

    def verify(self, payload: bytes) -> Tuple[E2EStatus, int]:
        """
        Verify incoming 8-byte CAN payload against E2E Profile 1 specifications.
        Returns (E2EStatus, received_counter).
        """
        if len(payload) != 8:
            return E2EStatus.INVALID_LENGTH, -1

        # Verify CRC
        expected_crc = calculate_crc8_sae_j1850(payload[:7], self.data_id)
        received_crc = payload[7]

        if expected_crc != received_crc:
            return E2EStatus.CRC_ERROR, payload[6] & 0x0F

        # Extract alive counter
        current_counter = payload[6] & 0x0F

        if self._rx_first_message:
            self._rx_first_message = False
            self._rx_last_counter = current_counter
            return E2EStatus.OK, current_counter

        # Calculate delta with 0..14 rollover
        if current_counter >= self._rx_last_counter:
            delta = current_counter - self._rx_last_counter
        else:
            delta = (current_counter + 15) - self._rx_last_counter

        self._rx_last_counter = current_counter

        if delta == 0:
            return E2EStatus.REPEATED, current_counter
        elif delta == 1:
            return E2EStatus.OK, current_counter
        elif 1 < delta <= self.max_allowed_delta:
            return E2EStatus.LOST_FRAME, current_counter
        else:
            return E2EStatus.SEQUENCE_ERROR, current_counter
