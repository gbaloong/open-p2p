"""Connect slow-plan predictions, latest-frame correction and the touch executor.

This serial reference implementation is NOT a GPU-priority scheduler. It is
usable with the trace backend for replay, with no screen-capture/device hooks.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .actions import MobileAction, TouchPacket, TouchMapper, acknowledged_action
from .models import HybridPrediction, FastCorrector
from .runtime import TouchExecutor


@dataclass
class ActionPlan:
    prediction: HybridPrediction
    observation_ns: int
    created_ns: int
    expires_ns: int
    version: int

    def validate(self):
        TouchPacket(self.version, self.observation_ns, self.created_ns, self.expires_ns, (), self.version)
        vector = self.prediction.vector()
        if vector.ndim != 2 or vector.shape[0] != 1 or not torch.isfinite(vector).all():
            raise ValueError("Plan must be a finite single-stream prediction")
        if (self.prediction.controls.abs() > 1).any():
            raise ValueError("Plan controls outside normalized range")


class CorrectedController:
    def __init__(self, corrector: FastCorrector, mapper: TouchMapper, executor: TouchExecutor,
                 *, hold_ns=50_000_000):
        if not 0 < hold_ns <= executor.max_hold:
            raise ValueError("Invalid control hold duration")
        if corrector.buttons != len(mapper.layout.button_positions):
            raise ValueError("Corrector/layout mismatch")
        self.corrector = corrector.eval()
        self.mapper, self.executor = mapper, executor
        self.plan = None
        self.hold_ns = hold_ns
        self.sequence = 0
        self.last_tick_ns = None
        self.actual = MobileAction(buttons=(False,)*corrector.buttons)

    def set_plan(self, plan: ActionPlan, now_ns: int) -> bool:
        plan.validate()
        if not plan.created_ns <= now_ns < plan.expires_ns:
            return False
        if self.plan and plan.version <= self.plan.version:
            return False
        prediction = HybridPrediction(plan.prediction.controls.detach().clone(),
                                      plan.prediction.button_logits.detach().clone())
        self.plan = ActionPlan(prediction, plan.observation_ns, plan.created_ns, plan.expires_ns, plan.version)
        return True

    @torch.inference_mode()
    def tick(self, frame: torch.Tensor, captured_ns: int, now_ns: int):
        if self.last_tick_ns is not None and now_ns < self.last_tick_ns:
            self.executor.stop()
            raise ValueError("Controller clock moved backwards")
        if self.plan is None or now_ns >= self.plan.expires_ns:
            self.executor.stop()
            return None
        if not self.plan.observation_ns <= captured_ns <= now_ns:
            raise ValueError("Frame is from the future or older than the current plan")
        if now_ns-captured_ns > self.executor.max_age:
            self.executor.stop()
            return None
        if not self.executor.armed:
            return None
        device = next(self.corrector.parameters()).device
        previous = torch.tensor([self.actual.vector()], device=device, dtype=torch.float32)
        timing = torch.tensor([[(now_ns-self.plan.observation_ns)/1e9,
                               0.0 if self.last_tick_ns is None else (now_ns-self.last_tick_ns)/1e9]],
                              device=device, dtype=torch.float32)
        plan = HybridPrediction(self.plan.prediction.controls.to(device), self.plan.prediction.button_logits.to(device))
        predicted = self.corrector(frame.to(device), plan, previous, timing)
        values = predicted.controls[0].float().cpu().tolist()
        buttons = tuple(bool(x) for x in (predicted.button_logits[0] >= 0).cpu().tolist())
        action = MobileAction(*values, buttons)
        before = self.executor.actual
        points = self.mapper.points(action, before)
        self.sequence = max(self.sequence+1, self.executor.last_sequence+1)
        packet = TouchPacket(self.sequence, captured_ns, now_ns,
                             min(now_ns+self.hold_ns, self.plan.expires_ns), points, self.plan.version)
        if not self.executor.submit(packet, now_ns):
            return None
        receipt = self.executor.tick(now_ns)
        if receipt is not None:
            self.actual = acknowledged_action(self.mapper.layout, before, receipt.points)
            self.last_tick_ns = receipt.acknowledged_ns
        return receipt
