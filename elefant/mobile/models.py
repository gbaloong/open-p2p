"""Parallel hybrid actions, time/goal conditioning, and latest-frame residual correction.

These are independently implemented experimental modules, NOT imported/trained
A2C2, OFT, RTC or ROCKET weights. The temporal inputs must be causal.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class HybridPrediction:
    controls: torch.Tensor
    button_logits: torch.Tensor

    def vector(self):
        return torch.cat((self.controls, self.button_logits.sigmoid()), dim=-1)


class HybridActionHead(nn.Module):
    def __init__(self, feature_dim: int, buttons: int = 8, hidden: int = 256):
        super().__init__()
        if feature_dim <= 0 or not 1 <= buttons <= 8 or hidden <= 0:
            raise ValueError("Invalid action head dimensions")
        self.buttons = buttons
        self.shared = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.SiLU())
        self.continuous = nn.Linear(hidden, 4)
        self.binary = nn.Linear(hidden, buttons)
        nn.init.constant_(self.binary.bias, -4.0)

    def forward(self, features) -> HybridPrediction:
        h = self.shared(features)
        return HybridPrediction(self.continuous(h).tanh(), self.binary(h))


def hybrid_loss(pred: HybridPrediction, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if targets.shape != pred.vector().shape or weight.shape != targets.shape[:-1]:
        raise ValueError("Action/weight shape mismatch")
    if not torch.isfinite(weight).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite training targets or weights")
    if (weight < 0).any() or (weight > 1).any():
        raise ValueError("Confidence must be in [0,1]")
    if (targets[..., :4].abs() > 1).any() or ((targets[..., 4:] != 0) & (targets[..., 4:] != 1)).any():
        raise ValueError("Controls must be normalized and button labels binary")
    continuous = F.smooth_l1_loss(pred.controls, targets[..., :4], reduction="none").mean(-1)
    buttons = F.binary_cross_entropy_with_logits(pred.button_logits, targets[..., 4:], reduction="none").mean(-1)
    # Return a differentiable ZERO if no frame is labelled. Never invent labels.
    return ((continuous + buttons)*weight).sum()/weight.sum().clamp_min(1)


class TimeGoalConditioner(nn.Module):
    def __init__(self, output_dim: int, goal_dim: int = 32):
        super().__init__()
        self.goal_dim = goal_dim
        self.net = nn.Sequential(nn.Linear(goal_dim+3, 128), nn.SiLU(), nn.Linear(128, output_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, timing, goal, visible):
        if timing.shape[-1] != 2 or goal.shape[-1] != self.goal_dim or visible.shape != timing.shape[:-1]:
            raise ValueError("Condition shapes do not match")
        if goal.shape[:-1] != timing.shape[:-1]:
            raise ValueError("Goal/time batch shapes do not match")
        if not torch.isfinite(timing).all() or (timing < 0).any() or not torch.isfinite(goal).all():
            raise ValueError("Conditions must be finite; times cannot be negative")
        # Hiding a goal must remove its entire embedding, not simply add a flag.
        if visible.dtype != torch.bool:
            raise ValueError("Goal visibility must be boolean")
        valid = visible.unsqueeze(-1)
        g = torch.where(valid, goal, torch.zeros_like(goal))
        return self.net(torch.cat((timing.clamp_max(10), g, valid.to(goal.dtype)), -1))


class FastCorrector(nn.Module):
    """Own visual path; zero-initialized output starts as identity on the main plan."""
    def __init__(self, buttons: int = 8, feature_dim: int = 128, max_residual: float = 0.25):
        super().__init__()
        if not 1 <= buttons <= 8 or not 0 < max_residual <= 1 or feature_dim < 8:
            raise ValueError("Invalid corrector configuration")
        self.buttons = buttons
        self.max_residual = max_residual
        self.vision = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, feature_dim, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.editor = nn.Sequential(nn.Linear(feature_dim+2*(4+buttons)+2, 256), nn.SiLU(),
                                    nn.Linear(256, 4+buttons))
        nn.init.zeros_(self.editor[-1].weight)
        nn.init.zeros_(self.editor[-1].bias)

    def forward(self, latest_frame, plan: HybridPrediction, executed, timing):
        if latest_frame.dtype != torch.uint8 or latest_frame.ndim != 4 or latest_frame.shape[1] != 3:
            raise ValueError("Expected latest RGB uint8 frame [B,3,H,W]")
        if timing.shape != (latest_frame.shape[0], 2) or not torch.isfinite(timing).all() or (timing < 0).any():
            raise ValueError("Timing must contain nonnegative plan age and step seconds")
        base = plan.vector()
        if executed.shape != base.shape or base.shape != (latest_frame.shape[0], 4+self.buttons):
            raise ValueError("Plan/executed action shape mismatch")
        if not torch.isfinite(base).all() or not torch.isfinite(executed).all():
            raise ValueError("Actions must be finite")
        image = latest_frame.to(dtype=next(self.parameters()).dtype)/255
        h = self.vision(image)
        delta = self.editor(torch.cat((h, base.to(h.dtype), executed.to(h.dtype), timing.to(h.dtype).clamp_max(10)), -1))
        controls = (plan.controls + self.max_residual*delta[..., :4].tanh()).clamp(-1, 1)
        # Discrete states are edited as logits; never add/subtract touch IDs.
        logits = plan.button_logits + delta[..., 4:].clamp(-5, 5)
        return HybridPrediction(controls, logits)
