import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
import pytest
import torch
from torch import nn
from elefant.mobile.policy import MobilePolicy
from elefant.mobile.models import hybrid_loss
from elefant.mobile.train_corrector import main as train_corrector
from elefant.mobile.pretrain_video import main as train_latent
from elefant.mobile.runtime import BackendAck, TouchExecutor, TraceBackend
from elefant.mobile.actions import TouchPacket, TouchPoint
from elefant.integration.benchmark import latency_summary


class FakeImageTokenizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(3, dim)
    def forward(self, x):
        return self.linear(x.mean((-1, -2))).unsqueeze(2)
    def get_n_img_tokens(self):
        return 1


class FakeCausalTransformer(nn.Module):
    """A cheap token-mask contract fixture, NOT a performance benchmark."""
    def __init__(self, steps, step_size, out_position):
        super().__init__()
        n = steps*step_size
        allow = torch.zeros(n, n)
        for q in range(n):
            qs = q//step_size
            for k in range(n):
                ks = k//step_size
                allow[q,k] = float(ks < qs or (ks == qs and k % step_size <= out_position))
        self.register_buffer("allow", allow/allow.sum(-1, keepdim=True))
    def forward(self, x):
        return self.allow @ x, None, {}, {}


class FakeCore(nn.Module):
    def __init__(self, dim=8, steps=3, n_actions=2):
        super().__init__()
        self.config = SimpleNamespace(embed_dim=dim, n_action_tokens=n_actions, n_steps=steps,
                                      n_thinking_tokens=1)
        self.image_tokenizer = FakeImageTokenizer(dim)
        self.img_pos_tokens = nn.Parameter(torch.randn(1, 1, dim))
        self.text_embedding_for_no_text_input = nn.Parameter(torch.randn(1, 1, dim))
        self.text_pos_tokens = nn.Parameter(torch.randn(1, 1, dim))
        self.thinking_pos_tokens = nn.Parameter(torch.randn(1, 1, dim))
        self.action_out_token = nn.Parameter(torch.randn(1, 1, dim))
        self.action_pos_tokens = nn.Parameter(torch.randn(1, n_actions, dim))
        self.action_decoder = nn.Linear(dim, dim)
        step_size = 4+n_actions
        self._transformer = FakeCausalTransformer(steps, step_size, 3)
        self.register_buffer("output_action_token_idx", torch.tensor([i*step_size+3 for i in range(steps)]))
        self.mask_moves = 0
    def block_mask_to_device(self, device):
        self.mask_moves += 1


def batch():
    return (torch.randint(256, (2,3,3,8,8), dtype=torch.uint8),
            torch.zeros(2,3,6), torch.zeros(2,3,2))


def test_mobile_policy_uses_core_and_has_trainable_heads():
    policy = MobilePolicy(FakeCore(), buttons=2, goal_dim=4)
    frames, executed, timing = batch()
    output = policy(frames, executed, timing)
    loss = hybrid_loss(output, torch.zeros(2,3,6), torch.ones(2,3))
    loss.backward()
    assert policy.core.image_tokenizer.linear.weight.grad is not None
    assert policy.action_encoder[-1].weight.grad is not None
    assert policy.head.binary.weight.grad is not None
    assert not any(p.requires_grad for p in policy.core.action_decoder.parameters())
    assert policy.parameter_report()["trainable"] < policy.parameter_report()["total"]
    policy(frames, executed, timing)
    assert policy.core.mask_moves == 1


def test_mobile_current_action_label_cannot_leak_to_current_output():
    policy = MobilePolicy(FakeCore(), buttons=2, goal_dim=4).eval()
    frames, executed, timing = batch()
    original = policy(frames, executed, timing).controls.detach()
    changed = executed.clone()
    changed[:,1] = 0.9
    modified = policy(frames, changed, timing).controls.detach()
    assert torch.equal(original[:,:2], modified[:,:2])
    assert not torch.equal(original[:,2], modified[:,2])


def test_future_frame_cannot_change_past_output():
    policy = MobilePolicy(FakeCore(), buttons=2, goal_dim=4).eval()
    frames, executed, timing = batch()
    original = policy(frames, executed, timing).controls.detach()
    frames[:,2] = 255-frames[:,2]
    modified = policy(frames, executed, timing).controls.detach()
    assert torch.equal(original[:,:2], modified[:,:2])


def test_actual_upstream_mask_contract_when_repository_is_present():
    source = Path(__file__).resolve().parents[2]/"elefant/policy_model/policy_transformer.py"
    if not source.is_file():
        pytest.skip("Full upstream source absent in isolated patch workspace; repository CI checks it")
    parsed = ast.parse(source.read_text())
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "_img_policy_causal_mask")
    namespace = {"Optional": Optional}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    mask = namespace["_img_policy_causal_mask"](0, 1, 1, 3, history_len=3, n_text_tokens=1)
    assert bool(mask(0,0,3,1))
    assert not bool(mask(0,0,3,4))
    assert not bool(mask(0,0,3,7))
    assert bool(mask(0,0,9,4))
    assert not bool(mask(0,0,9,3))


def test_late_backend_ack_triggers_release():
    class Slow(TraceBackend):
        def apply(self, events, now_ns):
            self.events.extend(events)
            return BackendAck(True, now_ns+200)
    backend = Slow()
    executor = TouchExecutor(backend)
    executor.submit(TouchPacket(1,0,0,100,(TouchPoint(0,.2,.2),)),0)
    assert executor.tick(0) is None
    assert not executor.armed and not executor.actual and backend.releases == 1


@pytest.mark.parametrize("times", [[], [-1], [float("nan")], [float("inf")]])
def test_latency_rejects_bad_samples(times):
    with pytest.raises(ValueError):
        latency_summary(times)


def test_latency_reports_all_samples():
    report = latency_summary([.01, .02, .03, .1])
    assert report["count"] == 4
    assert report["max_ms"] == 100
    assert report["p95_ms"] > 30


def write_corrections(path, recording_id):
    n = 4
    np.savez(path, frames=np.zeros((n,3,16,16), np.uint8), plan_controls=np.zeros((n,4), np.float32),
             plan_logits=np.zeros((n,2), np.float32), executed=np.zeros((n,6),np.float32),
             targets=np.ones((n,6),np.float32), timing=np.zeros((n,2),np.float32),
             weight=np.ones(n,np.float32), recording_ids=np.full(n,recording_id))


def test_corrector_training_cli_writes_real_updated_checkpoint(tmp_path):
    train, valid = tmp_path/"train.npz", tmp_path/"valid.npz"
    write_corrections(train,"a")
    write_corrections(valid,"b")
    out = tmp_path/"out"
    train_corrector(["--train",str(train),"--validation",str(valid),"--output",str(out),
                     "--buttons","2","--feature-dim","16","--steps","3","--batch-size","2","--device","cpu"])
    saved = torch.load(out/"corrector.ckpt", weights_only=True)
    assert saved["state_dict"]["editor.2.bias"].abs().sum() > 0
    report = json.loads((out/"training.json").read_text())
    assert report["steps"] == 3 and np.isfinite(report["validation_loss"])
    assert report["gameplay_verified"] is False


def test_corrector_training_refuses_recording_leakage(tmp_path):
    a,b=tmp_path/"a.npz",tmp_path/"b.npz"
    write_corrections(a,"same")
    write_corrections(b,"same")
    with pytest.raises(ValueError, match="leakage"):
        train_corrector(["--train",str(a),"--validation",str(b),"--buttons","2","--device","cpu"])


def test_offline_latent_training_does_not_cross_episodes(tmp_path):
    file = tmp_path/"features.npz"
    np.savez(file, features=np.random.default_rng(2).normal(size=(6,8)).astype(np.float32),
             episode_ids=np.array(["a","a","a","b","b","b"]))
    out = tmp_path/"out"
    train_latent(["--features",str(file),"--output",str(out),"--steps","2","--device","cpu"])
    report=json.loads((out/"training.json").read_text())
    assert report["within_episode_pairs"] == 4
    assert report["not_touch_labels"] is True
    assert (out/"latents.ckpt").is_file()


@pytest.mark.parametrize("module", ["elefant.mobile.train", "elefant.mobile.train_corrector",
                                     "elefant.mobile.pretrain_video", "elefant.integration.benchmark",
                                     "elefant.policy_model.train"])
def test_cli_help_does_not_load_gpu_or_full_upstream_dependencies(module):
    result=subprocess.run([sys.executable,"-m",module,"--help"],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
