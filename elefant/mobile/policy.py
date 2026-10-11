"""An OpenP2P-core adapter with parallel mobile actions and causal action history.

This is a full-window TRAINING/reference-inference path, not a claimed optimized
mobile KV-cache implementation. The upstream PC policy remains independently runnable.
"""
from __future__ import annotations
import torch
from torch import nn
from .models import HybridActionHead, TimeGoalConditioner


class MobilePolicy(nn.Module):
    def __init__(self, core: nn.Module, buttons: int = 8, goal_dim: int = 32):
        super().__init__()
        self.core = core
        self.buttons = buttons
        self.goal_dim = goal_dim
        d = core.config.embed_dim
        n = core.config.n_action_tokens
        self.action_encoder = nn.Sequential(nn.Linear(4+buttons, 128), nn.SiLU(), nn.Linear(128, n*d))
        self.conditioner = TimeGoalConditioner(d, goal_dim)
        self.head = HybridActionHead(d, buttons)
        # The pretrained PC autoregressive decoder is not a mobile action head.
        # Preserve it for checkpoint traceability, but explicitly exclude it.
        self.core.action_decoder.requires_grad_(False)
        self._mask_device = None

    @classmethod
    def from_pretrained_policy(cls, policy, **kwargs):
        return cls(policy.bc_transformer, **kwargs)

    def forward(self, frames, executed, timing, goal=None, visible=None):
        if frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be RGB uint8 [B,T,3,H,W]")
        b, t = frames.shape[:2]
        if t != self.core.config.n_steps or executed.shape != (b, t, 4+self.buttons):
            raise ValueError("Window length/action dimension mismatch")
        if timing.shape != (b, t, 2):
            raise ValueError("timing must be [B,T,2] age and elapsed seconds")
        if not torch.isfinite(executed).all():
            raise ValueError("Actual action history must be finite")
        if self._mask_device != str(frames.device):
            self.core.block_mask_to_device(frames.device)
            self._mask_device = str(frames.device)
        d = self.core.config.embed_dim
        n = self.core.config.n_action_tokens
        if goal is None:
            goal = torch.zeros(b, t, self.goal_dim, device=frames.device)
        if visible is None:
            visible = torch.zeros(b, t, device=frames.device, dtype=torch.bool)
        condition = self.conditioner(timing, goal, visible)
        images = self.core.image_tokenizer(frames.float()/255)
        images = images + self.core.img_pos_tokens.unsqueeze(0)
        # No extra Gemma model is loaded. Use the pretrained 'no text' token.
        text = self.core.text_embedding_for_no_text_input + self.core.text_pos_tokens
        text = text.unsqueeze(0).expand(b, t, -1, -1)
        thinking = self.core.thinking_pos_tokens.unsqueeze(0).expand(b, t, -1, -1)
        output = self.core.action_out_token.unsqueeze(0).expand(b, t, -1, -1)
        output = output + condition.unsqueeze(2)
        actions = self.action_encoder(executed).reshape(b, t, n, d)
        actions = actions + self.core.action_pos_tokens.unsqueeze(0)
        packed = torch.cat((text, images, thinking, output, actions), dim=2).flatten(1, 2)
        # The EXISTING OpenP2P mask hides all current and future action tokens
        # from the current output token. Do not pre-shift executed[] a second time.
        hidden, _, _, _ = self.core._transformer(packed)
        features = hidden.index_select(1, self.core.output_action_token_idx)
        return self.head(features)

    def parameter_report(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "excluded": "unused pretrained PC autoregressive decoder; mobile head replaces it"}
