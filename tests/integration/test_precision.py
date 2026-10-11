"""Regression: full bf16 models must not receive hard-coded fp32 frame inputs."""
import torch
from elefant.mobile.policy import MobilePolicy
from test_policy_and_tools import FakeCore, batch


def test_mobile_bf16_true_forward_without_autocast():
    policy = MobilePolicy(FakeCore(), buttons=2, goal_dim=4).to(dtype=torch.bfloat16)
    frames, executed, timing = batch()
    pred = policy(frames, executed, timing)
    assert pred.controls.dtype == torch.bfloat16
    assert torch.isfinite(pred.controls).all()
    pred.controls.float().sum().backward()
    assert policy.head.continuous.weight.grad is not None


def test_mobile_float32_still_supported():
    policy = MobilePolicy(FakeCore(), buttons=2, goal_dim=4)
    frames, executed, timing = batch()
    pred = policy(frames, executed, timing)
    assert pred.controls.dtype == torch.float32
