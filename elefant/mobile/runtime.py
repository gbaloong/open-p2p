"""Latest-only execution, monotonic deadlines, acknowledgements and fail-closed stop.

The included backend is a TRACE simulator, not an Android driver. A real driver
must acknowledge completion and implement release_all without anti-cheat bypasses.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Protocol
import threading

from .actions import TouchPacket, TouchPoint, _integer


@dataclass(frozen=True)
class TouchEvent:
    kind: str
    point: TouchPoint


@dataclass(frozen=True)
class BackendAck:
    ok: bool
    completed_ns: int


@dataclass(frozen=True)
class Receipt:
    sequence: int
    acknowledged_ns: int
    points: tuple[TouchPoint, ...]
    observation_ns: int


class TouchBackend(Protocol):
    def apply(self, events: tuple[TouchEvent, ...], now_ns: int) -> BackendAck: ...
    def release_all(self) -> bool: ...


class TraceBackend:
    def __init__(self):
        self.events: list[TouchEvent] = []
        self.releases = 0
    def apply(self, events, now_ns):
        self.events.extend(events)
        return BackendAck(True, now_ns)
    def release_all(self):
        self.releases += 1
        return True


class TouchExecutor:
    def __init__(self, backend: TouchBackend, *, max_observation_age_ns=100_000_000,
                 max_hold_ns=250_000_000):
        _integer(max_observation_age_ns, "max_observation_age_ns")
        _integer(max_hold_ns, "max_hold_ns")
        if not max_observation_age_ns or not max_hold_ns:
            raise ValueError("Limits must be positive")
        self.backend = backend
        self.max_age = max_observation_age_ns
        self.max_hold = max_hold_ns
        self.actual: tuple[TouchPoint, ...] = ()
        self.pending: TouchPacket | None = None
        self.current: TouchPacket | None = None
        self.last_sequence = -1
        self.plan_version = -1
        self.armed = True
        self.release_confirmed = True
        self.receipts = deque(maxlen=2048)
        self._last_now = 0
        self._lock = threading.RLock()

    def _clock(self, now_ns):
        _integer(now_ns, "now_ns")
        if now_ns < self._last_now:
            self.stop()
            raise ValueError("Clock moved backwards; executor stopped")
        self._last_now = now_ns

    def submit(self, packet: TouchPacket, now_ns: int) -> bool:
        with self._lock:
            self._clock(now_ns)
            if not self.armed:
                return False
            if not packet.observation_ns <= packet.created_ns <= now_ns < packet.expires_ns:
                return False
            if now_ns-packet.observation_ns > self.max_age:
                return False
            if packet.expires_ns-packet.created_ns > self.max_hold:
                return False
            if packet.sequence <= self.last_sequence or packet.plan_version < self.plan_version:
                return False
            self.pending = packet
            self.last_sequence = packet.sequence
            self.plan_version = packet.plan_version
            return True

    def _release(self):
        try:
            success = self.backend.release_all() is True
        except Exception:
            success = False
        self.release_confirmed = success
        if success:
            self.actual = ()
        else:
            self.armed = False  # Actual device state is UNKNOWN, not silently empty.
        self.current = None
        self.pending = None
        return success

    def stop(self) -> bool:
        with self._lock:
            self.armed = False
            return self._release()

    def rearm(self) -> bool:
        with self._lock:
            if not self._release():
                return False
            # Sequence numbers remain monotonic across rearm; replay is rejected.
            self.armed = True
            return True

    def tick(self, now_ns: int) -> Receipt | None:
        """now_ns is an injected common-clock time, not a game-server acknowledgement."""
        with self._lock:
            self._clock(now_ns)
            if not self.armed:
                return None
            packet = self.pending
            if packet is not None and (now_ns >= packet.expires_ns or
                                       now_ns-packet.observation_ns > self.max_age):
                self.pending = packet = None
            if packet is None:
                if self.current and (now_ns >= self.current.expires_ns or
                                     now_ns-self.current.observation_ns > self.max_age):
                    self._release()
                return None
            old = {p.pointer_id: p for p in self.actual}
            new = {p.pointer_id: p for p in packet.points}
            events = [TouchEvent("up", old[k]) for k in sorted(old.keys()-new.keys())]
            events += [TouchEvent("move", new[k]) for k in sorted(old.keys() & new.keys()) if old[k] != new[k]]
            events += [TouchEvent("down", new[k]) for k in sorted(new.keys()-old.keys())]
            try:
                ack = self.backend.apply(tuple(events), now_ns)
                accepted = (isinstance(ack, BackendAck) and ack.ok is True
                            and type(ack.completed_ns) is int and ack.completed_ns >= now_ns)
            except Exception:
                accepted = False
            if not accepted:
                self.stop()
                raise RuntimeError("Backend did not acknowledge; stopped and attempted release_all")
            self.actual = packet.points
            self._last_now = ack.completed_ns
            if ack.completed_ns >= packet.expires_ns or ack.completed_ns-packet.observation_ns > self.max_age:
                self.stop()
                return None
            self.current = packet
            self.pending = None
            self.release_confirmed = not bool(self.actual)
            receipt = Receipt(packet.sequence, ack.completed_ns, self.actual, packet.observation_ns)
            self.receipts.append(receipt)
            return receipt
