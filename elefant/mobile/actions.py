"""Normalized mobile actions and a layout-specific multi-touch mapper.

Coordinates are normalized to the GAME viewport, not the encoded video including bars.
This module never injects inputs or assumes a particular game's button layout.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def _finite(value: float, lo: float, hi: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be finite in [{lo}, {hi}]")


def _integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class MobileAction:
    move_x: float = 0.0
    move_y: float = 0.0
    look_x: float = 0.0
    look_y: float = 0.0
    buttons: tuple[bool, ...] = ()

    def __post_init__(self):
        for name in ("move_x", "move_y", "look_x", "look_y"):
            _finite(getattr(self, name), -1.0, 1.0, name)
        if not isinstance(self.buttons, tuple) or len(self.buttons) > 8:
            raise ValueError("buttons must be a tuple with at most eight entries")
        if any(type(b) is not bool for b in self.buttons):
            raise ValueError("button states must be booleans, not probabilities")

    def vector(self) -> tuple[float, ...]:
        return (self.move_x, self.move_y, self.look_x, self.look_y, *map(float, self.buttons))


@dataclass(frozen=True)
class TouchPoint:
    pointer_id: int
    x: float
    y: float

    def __post_init__(self):
        _integer(self.pointer_id, "pointer_id")
        if self.pointer_id > 9:
            raise ValueError("At most ten touch slots are supported")
        _finite(self.x, 0, 1, "x")
        _finite(self.y, 0, 1, "y")


@dataclass(frozen=True)
class TouchPacket:
    sequence: int
    observation_ns: int
    created_ns: int
    expires_ns: int
    points: tuple[TouchPoint, ...]
    plan_version: int = 0

    def __post_init__(self):
        for name in ("sequence", "observation_ns", "created_ns", "expires_ns", "plan_version"):
            _integer(getattr(self, name), name)
        if not self.observation_ns <= self.created_ns < self.expires_ns:
            raise ValueError("Require observation <= creation < expiry")
        if not isinstance(self.points, tuple) or not all(isinstance(p, TouchPoint) for p in self.points):
            raise ValueError("points must be a tuple of TouchPoint")
        if len({p.pointer_id for p in self.points}) != len(self.points):
            raise ValueError("Duplicate touch slots")


@dataclass(frozen=True)
class TouchLayout:
    move_center: tuple[float, float]
    move_radius: float
    look_center: tuple[float, float]
    look_step: float
    button_positions: tuple[tuple[float, float], ...]

    def __post_init__(self):
        _finite(self.move_radius, 0.001, 0.25, "move_radius")
        _finite(self.look_step, 0.001, 0.1, "look_step")
        if len(self.button_positions) > 8:
            raise ValueError("Too many button slots")
        for xy in (self.move_center, self.look_center, *self.button_positions):
            if len(xy) != 2:
                raise ValueError("Each position needs x,y")
            for v in xy:
                _finite(v, 0, 1, "layout coordinate")
        if any(not self.move_radius <= v <= 1-self.move_radius for v in self.move_center):
            raise ValueError("Joystick radius crosses the viewport boundary")


class TouchMapper:
    """Map one control step against ACKNOWLEDGED contacts, never against proposals.

Look controls are normalized deltas PER CONTROL STEP. When the look pointer
reaches an edge, release it for one step before recentering. Training data must
use the same time step or explicitly rescale deltas. Layout must be calibrated.
"""
    def __init__(self, layout: TouchLayout):
        self.layout = layout

    def points(self, action: MobileAction, acknowledged: tuple[TouchPoint, ...] = ()) -> tuple[TouchPoint, ...]:
        if len(action.buttons) != len(self.layout.button_positions):
            raise ValueError("Action/button layout dimension mismatch")
        result = []
        if math.hypot(action.move_x, action.move_y) > 1e-6:
            norm = max(1.0, math.hypot(action.move_x, action.move_y))
            result.append(TouchPoint(0, self.layout.move_center[0] + self.layout.move_radius*action.move_x/norm,
                                    self.layout.move_center[1] + self.layout.move_radius*action.move_y/norm))
        if abs(action.look_x) + abs(action.look_y) > 1e-6:
            prior = next((p for p in acknowledged if p.pointer_id == 1), None)
            # First establish an anchor: moving on DOWN is not a swipe.
            if prior is None:
                result.append(TouchPoint(1, *self.layout.look_center))
            else:
                x = prior.x + self.layout.look_step*action.look_x
                y = prior.y + self.layout.look_step*action.look_y
                if 0.01 <= x <= 0.99 and 0.01 <= y <= 0.99:
                    result.append(TouchPoint(1, x, y))
        for i, (held, xy) in enumerate(zip(action.buttons, self.layout.button_positions)):
            if held:
                result.append(TouchPoint(i+2, *xy))
        return tuple(result)


def acknowledged_action(layout: TouchLayout, before: tuple[TouchPoint, ...],
                        after: tuple[TouchPoint, ...]) -> MobileAction:
    """Reconstruct ACKNOWLEDGED input, including zero motion during look re-anchoring."""
    old, new = {p.pointer_id: p for p in before}, {p.pointer_id: p for p in after}
    move = new.get(0)
    mx = (move.x-layout.move_center[0])/layout.move_radius if move else 0.0
    my = (move.y-layout.move_center[1])/layout.move_radius if move else 0.0
    lx = (new[1].x-old[1].x)/layout.look_step if 1 in old and 1 in new else 0.0
    ly = (new[1].y-old[1].y)/layout.look_step if 1 in old and 1 in new else 0.0
    clamp = lambda v: max(-1.0, min(1.0, v))
    return MobileAction(clamp(mx), clamp(my), clamp(lx), clamp(ly),
                        tuple(i+2 in new for i in range(len(layout.button_positions))))
