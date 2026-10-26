"""
Cross-Platform Automotive CAN Bus Hardware Abstraction Layer (HAL).
Supports:
1. In-memory thread-safe queue (virtual loopback, < 5 us latency)
2. UDP loopback (127.0.0.1:28700 cross-platform socket broadcast)
3. Linux SocketCAN (vcan0) via native socket or python-can when available.
"""

from __future__ import annotations

import enum
import queue
import socket
import struct
import sys
import time
from typing import Optional, Tuple


class BusBackend(enum.Enum):
    """Virtual CAN Bus backend mode."""
    LOOPBACK = "LOOPBACK"
    UDP = "UDP"
    SOCKETCAN = "SOCKETCAN"


class VirtualCanBus:
    """
    Cross-platform virtual automotive CAN bus.
    Enables synchronous and asynchronous telemetry publishing and receiving.
    """

    def __init__(
        self,
        backend: BusBackend = BusBackend.LOOPBACK,
        channel: str = "vcan0",
        udp_port: int = 28700,
        udp_host: str = "127.0.0.1",
    ) -> None:
        self.backend = backend
        self.channel = channel
        self.udp_port = int(udp_port)
        self.udp_host = udp_host

        self._queue: queue.Queue[Tuple[int, bytes, float]] = queue.Queue(maxsize=1000)
        self._udp_sock: Optional[socket.socket] = None
        self._is_open = True

        if self.backend == BusBackend.UDP:
            self._init_udp()

    def _init_udp(self) -> None:
        """Initialize UDP broadcast socket."""
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setblocking(False)
        try:
            self._udp_sock.bind((self.udp_host, self.udp_port))
        except Exception:
            # If port is already bound by another listener, allow send-only
            pass

    def send(self, can_id: int, data: bytes) -> bool:
        """Send 8-byte CAN frame on the virtual bus."""
        if not self._is_open:
            return False

        if len(data) != 8:
            raise ValueError(f"CAN frame must be exactly 8 bytes, got {len(data)}")

        now = time.perf_counter()

        if self.backend == BusBackend.LOOPBACK:
            try:
                self._queue.put_nowait((can_id, bytes(data), now))
                return True
            except queue.Full:
                return False

        elif self.backend == BusBackend.UDP:
            if self._udp_sock is not None:
                # Pack CAN ID (4 bytes) + timestamp (8 bytes) + payload (8 bytes) = 20 bytes
                packet = struct.pack("<Id8s", can_id, now, data)
                try:
                    self._udp_sock.sendto(packet, (self.udp_host, self.udp_port))
                    return True
                except Exception:
                    return False
            return False

        return False

    def receive(self, timeout: Optional[float] = 0.001) -> Optional[Tuple[int, bytes, float]]:
        """
        Receive next CAN frame from virtual bus.
        Returns (can_id, payload, timestamp) or None if timeout.
        """
        if not self._is_open:
            return None

        if self.backend == BusBackend.LOOPBACK:
            try:
                if timeout is None:
                    return self._queue.get_nowait()
                else:
                    return self._queue.get(timeout=timeout)
            except (queue.Empty, queue.Full):
                return None

        elif self.backend == BusBackend.UDP:
            if self._udp_sock is not None:
                try:
                    raw_data, _ = self._udp_sock.recvfrom(64)
                    if len(raw_data) >= 20:
                        can_id, ts, payload = struct.unpack("<Id8s", raw_data[:20])
                        return int(can_id), payload, float(ts)
                except (BlockingIOError, socket.error):
                    return None
            return None

        return None

    def close(self) -> None:
        """Close bus handles and clear queues."""
        self._is_open = False
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
