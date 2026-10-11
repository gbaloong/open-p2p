"""Bounded visual-goal memory; a remembered goal is not a currently visible goal."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from .actions import _integer


@dataclass(frozen=True)
class VisualGoal:
    key: str
    features: tuple[float, ...]
    last_seen_ns: int
    confidence: float
    visible: bool = True

    def __post_init__(self):
        _integer(self.last_seen_ns, "last_seen_ns")
        if not self.key or not self.features or not all(math.isfinite(v) for v in self.features):
            raise ValueError("Invalid visual goal")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence outside [0,1]")
        if type(self.visible) is not bool:
            raise ValueError("visible must be bool")


class GoalMemory:
    def __init__(self, *, ttl_ns=3_000_000_000, capacity=64):
        _integer(ttl_ns, "ttl_ns")
        _integer(capacity, "capacity")
        if not ttl_ns or not capacity:
            raise ValueError("TTL and capacity must be positive")
        self.ttl_ns, self.capacity = ttl_ns, capacity
        self.entries: dict[str, VisualGoal] = {}
        self.frame_ns = -1

    def observe(self, frame_ns: int, observations: list[VisualGoal]):
        _integer(frame_ns, "frame_ns")
        if frame_ns <= self.frame_ns:
            raise ValueError("Out-of-order goal observation")
        if len({g.key for g in observations}) != len(observations):
            raise ValueError("Duplicate goal ids")
        if any(g.last_seen_ns != frame_ns or not g.visible for g in observations):
            raise ValueError("Observations must be confirmed visible in this frame")
        updated = {k: replace(g, visible=False) for k, g in self.entries.items()
                   if frame_ns-g.last_seen_ns <= self.ttl_ns}
        updated.update({g.key: g for g in observations})
        ordered = sorted(updated.values(), key=lambda g: (g.last_seen_ns, g.confidence), reverse=True)
        self.entries = {g.key: g for g in ordered[:self.capacity]}
        self.frame_ns = frame_ns

    def get(self, key: str, now_ns: int, *, require_visible=True) -> VisualGoal | None:
        _integer(now_ns, "now_ns")
        if now_ns < self.frame_ns:
            raise ValueError("Memory query precedes the latest observation")
        goal = self.entries.get(key)
        if goal is None or now_ns-goal.last_seen_ns > self.ttl_ns:
            return None
        if require_visible and not goal.visible:
            return None
        return goal
