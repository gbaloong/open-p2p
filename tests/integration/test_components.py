import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from elefant.integration.checkpoints import initialize_weights, read_weights, sha256_file, write_json_atomic
from elefant.integration.training import validate_launch, prepare_config
from elefant.mobile.actions import MobileAction, TouchPoint, TouchPacket, TouchLayout, TouchMapper, acknowledged_action
from elefant.mobile.runtime import TraceBackend, TouchExecutor
from elefant.mobile.models import HybridActionHead, HybridPrediction, FastCorrector, hybrid_loss, TimeGoalConditioner
from elefant.mobile.controller import CorrectedController, ActionPlan
from elefant.mobile.data import read_manifest, MobileDataset
from elefant.mobile.memory import GoalMemory, VisualGoal
from elefant.mobile.video_learning import LatentActionVQ, skill_segments

torch.set_num_threads(2)


def packet(seq=1, obs=0, now=0, expiry=100, points=(TouchPoint(0, 0.2, 0.3),), version=0):
    return TouchPacket(seq, obs, now, expiry, points, version)


def layout(buttons=2):
    return TouchLayout((0.2, 0.8), 0.1, (0.7, 0.5), 0.02, tuple((0.9, 0.1+i*0.1) for i in range(buttons)))


def test_checkpoint_roundtrip_and_no_optimizer_restore(tmp_path):
    source, target = nn.Linear(3, 2), nn.Linear(3, 2)
    p = tmp_path / "model.ckpt"
    torch.save({"state_dict": source.state_dict(), "optimizer_states": [{"ignored": 1}]}, p)
    report = initialize_weights(target, p)
    assert report["mode"] == "weights_only_initialization"
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in source.state_dict().items())
    assert len(sha256_file(p)) == 64


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape"])
def test_checkpoint_mismatch_fails_before_mutation(tmp_path, problem):
    model = nn.Linear(3, 2)
    original = copy.deepcopy(model.state_dict())
    changed = {k: torch.ones_like(v) for k, v in original.items()}
    if problem == "missing":
        del changed["bias"]
    elif problem == "unexpected":
        changed["other"] = torch.ones(1)
    else:
        changed["bias"] = torch.ones(7)
    p = tmp_path / "bad.ckpt"
    torch.save(changed, p)
    with pytest.raises(ValueError):
        initialize_weights(model, p)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


def test_checkpoint_compile_prefix_collision(tmp_path):
    p = tmp_path / "duplicate.ckpt"
    torch.save({"x": torch.zeros(1), "_orig_mod.x": torch.ones(1)}, p)
    with pytest.raises(ValueError, match="Colliding"):
        read_weights(p)


def test_checkpoint_compiled_key_supported(tmp_path):
    m = nn.Linear(1, 1)
    p = tmp_path / "compiled.ckpt"
    torch.save({"_orig_mod."+k: v for k, v in m.state_dict().items()}, p)
    initialize_weights(m, p)


def test_atomic_report_rejects_nan_without_clobber(tmp_path):
    p = tmp_path / "report.json"
    write_json_atomic(p, {"ok": True})
    with pytest.raises(ValueError):
        write_json_atomic(p, {"bad": float("nan")})
    assert json.loads(p.read_text()) == {"ok": True}
    assert list(tmp_path.iterdir()) == [p]


def launch(**changes):
    d = dict(init_checkpoint="source.ckpt", resume=None, from_scratch=False, accumulate=4,
             sequence_length=8, max_steps=10, output="outputs/test")
    return SimpleNamespace(**(d | changes))


@pytest.mark.parametrize("changes", [dict(resume="x"), dict(init_checkpoint=None), dict(accumulate=0),
                                    dict(sequence_length=0), dict(max_steps=-1)])
def test_invalid_training_options(changes):
    with pytest.raises(ValueError):
        validate_launch(launch(**changes))


def test_multi_rank_guard(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process"):
        validate_launch(launch())


def test_training_config_does_not_mutate_original():
    cfg = SimpleNamespace(shared=SimpleNamespace(output_path="orig", n_seq_timesteps=200),
                          policy_model=SimpleNamespace(attention_history_len=[200]*22),
                          stage3_finetune=SimpleNamespace(n_training_steps=100, accumulate_grad_batches=1,
                                                          freeze_transformer_layers_for_steps=20))
    cfg.model_copy = lambda deep: copy.deepcopy(cfg)
    result = prepare_config(cfg, launch())
    assert cfg.shared.n_seq_timesteps == 200
    assert result.shared.n_seq_timesteps == max(result.policy_model.attention_history_len) == 8
    assert result.stage3_finetune.accumulate_grad_batches == 1
    assert result.stage3_finetune.freeze_transformer_layers_for_steps == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, 1.01, True, "0"])
def test_controls_reject_invalid(value):
    with pytest.raises(ValueError):
        MobileAction(move_x=value)


@pytest.mark.parametrize("pointer", [-1, 10, True, 1.2])
def test_pointer_ids(pointer):
    with pytest.raises(ValueError):
        TouchPoint(pointer, 0.5, 0.5)


def test_packet_duplicate_and_bad_time():
    with pytest.raises(ValueError):
        packet(points=(TouchPoint(0, 0, 0), TouchPoint(0, 1, 1)))
    with pytest.raises(ValueError):
        packet(obs=2, now=1)
    with pytest.raises(ValueError):
        packet(expiry=0)


def test_mapper_multi_touch_anchor_feedback():
    mapper = TouchMapper(layout())
    action = MobileAction(1, 1, 0.5, 0.2, (True, False))
    first = mapper.points(action)
    assert {p.pointer_id for p in first} == {0, 1, 2}
    actual = acknowledged_action(mapper.layout, (), first)
    assert actual.look_x == actual.look_y == 0
    assert actual.move_x == pytest.approx(2**-0.5)
    second = mapper.points(action, first)
    actual = acknowledged_action(mapper.layout, first, second)
    assert actual.look_x == pytest.approx(0.5)
    assert actual.look_y == pytest.approx(0.2)


def test_mapper_releases_look_at_edge():
    mapper = TouchMapper(layout())
    result = mapper.points(MobileAction(look_x=1, buttons=(False, False)), (TouchPoint(1, 0.99, 0.5),))
    assert not any(p.pointer_id == 1 for p in result)


def test_latest_packet_wins_and_replay_rejected():
    backend = TraceBackend()
    executor = TouchExecutor(backend, max_observation_age_ns=100, max_hold_ns=100)
    assert executor.submit(packet(), 0)
    assert executor.submit(packet(seq=2, points=(TouchPoint(2, 0.8, 0.8),)), 0)
    assert not executor.submit(packet(), 0)
    receipt = executor.tick(0)
    assert receipt.sequence == 2
    assert len(backend.events) == 1 and backend.events[0].point.pointer_id == 2


def test_executor_event_order_and_expiry():
    backend = TraceBackend()
    executor = TouchExecutor(backend, max_observation_age_ns=100, max_hold_ns=100)
    executor.submit(packet(), 0)
    executor.tick(0)
    executor.submit(packet(seq=2, points=(TouchPoint(2, 0.4, 0.5),)), 1)
    executor.tick(1)
    assert [e.kind for e in backend.events] == ["down", "up", "down"]
    executor.tick(100)
    assert executor.actual == () and backend.releases == 1


def test_executor_stale_future_and_old_plan():
    executor = TouchExecutor(TraceBackend(), max_observation_age_ns=5, max_hold_ns=100)
    assert not executor.submit(packet(), 6)
    assert not executor.submit(packet(seq=2, obs=8, now=8), 7)
    assert executor.submit(packet(seq=3, obs=8, now=8, version=3), 8)
    assert not executor.submit(packet(seq=4, obs=8, now=8, version=2), 8)


def test_expired_pending_never_executes():
    backend = TraceBackend()
    executor = TouchExecutor(backend)
    executor.submit(packet(expiry=3), 0)
    assert executor.tick(3) is None
    assert not backend.events


def test_backend_failure_disarms_and_requires_rearm():
    class Failure(TraceBackend):
        def apply(self, events, now_ns):
            return False
    backend = Failure()
    executor = TouchExecutor(backend)
    executor.submit(packet(), 0)
    with pytest.raises(RuntimeError):
        executor.tick(0)
    assert not executor.armed and backend.releases == 1
    assert not executor.submit(packet(seq=2), 1)
    assert executor.rearm()


def test_release_failure_keeps_actual_unknown():
    class Failure(TraceBackend):
        def release_all(self):
            return False
    executor = TouchExecutor(Failure())
    executor.submit(packet(), 0)
    executor.tick(0)
    assert not executor.stop()
    assert executor.actual and not executor.release_confirmed and not executor.armed


def test_clock_backwards_stops_executor():
    executor = TouchExecutor(TraceBackend())
    executor.tick(10)
    with pytest.raises(ValueError):
        executor.tick(9)
    assert not executor.armed


def test_head_backward_and_masked_loss():
    head = HybridActionHead(16, buttons=2)
    x = torch.randn(2, 3, 16)
    pred = head(x)
    assert pred.controls.shape == (2, 3, 4)
    labels = torch.zeros(2, 3, 6)
    weights = torch.ones(2, 3)
    loss = hybrid_loss(pred, labels, weights)
    loss.backward()
    assert all(p.grad is not None for p in head.parameters())
    zero = hybrid_loss(head(x), labels, torch.zeros_like(weights))
    assert zero.item() == 0 and zero.requires_grad


def test_head_can_fit_small_synthetic_batch():
    torch.manual_seed(7)
    head = HybridActionHead(8, buttons=2, hidden=16)
    optim = torch.optim.Adam(head.parameters(), lr=0.02)
    x, labels = torch.randn(4, 8), torch.zeros(4, 6)
    losses = []
    for _ in range(12):
        loss = hybrid_loss(head(x), labels, torch.ones(4))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_corrector_identity_then_backprop():
    model = FastCorrector(buttons=2, feature_dim=16)
    image = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    plan = HybridPrediction(torch.zeros(2, 4), torch.zeros(2, 2))
    result = model(image, plan, torch.zeros(2, 6), torch.zeros(2, 2))
    assert torch.equal(result.controls, plan.controls)
    assert torch.equal(result.button_logits, plan.button_logits)
    labels = torch.ones(2, 6)
    loss = hybrid_loss(result, labels, torch.ones(2))
    loss.backward()
    assert model.editor[-1].weight.grad.abs().sum() > 0


def test_time_goal_condition_masks_invisible_embedding():
    module = TimeGoalConditioner(16, goal_dim=4)
    nn.init.normal_(module.net[-1].weight)
    timing = torch.ones(2, 3, 2)
    invisible = torch.zeros(2, 3, dtype=torch.bool)
    a = module(timing, torch.randn(2, 3, 4), invisible)
    b = module(timing, torch.randn(2, 3, 4), invisible)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        module(-timing, torch.zeros(2, 3, 4), invisible)


def test_controller_connects_prediction_to_acknowledged_actions():
    corrector = FastCorrector(buttons=2, feature_dim=16)
    executor = TouchExecutor(TraceBackend())
    controller = CorrectedController(corrector, TouchMapper(layout()), executor)
    plan = ActionPlan(HybridPrediction(torch.tensor([[0.3, 0., 0.5, 0.]]), torch.tensor([[5., -5.]])),
                      0, 0, 200_000_000, 1)
    assert controller.set_plan(plan, 0)
    assert not controller.set_plan(plan, 0)
    frame = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
    assert controller.tick(frame, 0, 0)
    assert controller.actual.look_x == 0
    assert controller.tick(frame, 10_000_000, 10_000_000)
    assert controller.actual.look_x == pytest.approx(0.5)
    assert controller.actual.buttons == (True, False)
    assert controller.tick(frame, 200_000_000, 200_000_000) is None
    assert not executor.armed and not executor.actual


def test_goal_memory_visibility_expiry_and_capacity():
    memory = GoalMemory(ttl_ns=10, capacity=1)
    memory.observe(1, [VisualGoal("door", (0.2, 0.3), 1, 0.9)])
    assert memory.get("door", 1)
    memory.observe(2, [])
    assert memory.get("door", 2) is None
    assert memory.get("door", 2, require_visible=False)
    assert memory.get("door", 12, require_visible=False) is None
    memory.observe(3, [VisualGoal("exit", (0.3, 0.4), 3, 0.8)])
    assert set(memory.entries) == {"exit"}
    with pytest.raises(ValueError):
        memory.observe(3, [])


def test_latent_action_quantization_backward():
    model = LatentActionVQ(8, latent_dim=4, codes=8)
    current, future = torch.randn(6, 8), torch.randn(6, 8)
    result = model(current, future)
    assert result["indices"].shape == (6,)
    result["loss"].backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.codebook.weight.grad.abs().sum() > 0


def test_skill_segmentation_respects_minimum_lengths():
    errors = np.zeros(40)
    errors[[3, 10, 12, 25, 38]] = 10
    assert skill_segments(errors, minimum_length=8) == [(0, 10), (10, 25), (25, 40)]
    assert skill_segments([]) == []


def episode(tmp_path, name="a", source="logged", split="train", recording=None, confidence=None):
    n = 8
    arrays = dict(frames=np.stack([np.full((3, 8, 8), i, np.uint8) for i in range(n)]),
                  executed=np.zeros((n, 6), np.float32), targets=np.zeros((n, 6), np.float32),
                  timestamps_ns=np.arange(n, dtype=np.int64)*50_000_000)
    if confidence is not None:
        arrays["confidence"] = np.asarray(confidence, dtype=np.float32)
    np.savez(tmp_path / (name+".npz"), **arrays)
    return {"episode_id": name, "recording_id": recording or name, "split": split,
            "path": name+".npz", "label_source": source}


def manifest(tmp_path, records):
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def test_dataset_causal_delay_alignment(tmp_path):
    path = manifest(tmp_path, [episode(tmp_path)])
    dataset = MobileDataset(path, split="train", steps=3, buttons=2, delay_frames=2)
    item = dataset[0]
    assert item["frames"][:, 0, 0, 0].tolist() == [0, 1, 2]
    assert torch.allclose(item["timing"][:, 0], torch.full((3,), 0.1))
    assert torch.allclose(item["timing"][:, 1], torch.full((3,), 0.05))


def test_dataset_rejects_recording_leakage(tmp_path):
    path = manifest(tmp_path, [episode(tmp_path, "a", recording="same"),
                               episode(tmp_path, "b", split="test", recording="same")])
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(path)


def test_pseudo_confidence_is_not_invented(tmp_path):
    path = manifest(tmp_path, [episode(tmp_path, source="pseudo")])
    with pytest.raises(ValueError, match="explicit confidence"):
        MobileDataset(path, split="train", steps=2, buttons=2)


def test_pseudo_threshold_and_unlabeled_exclusion(tmp_path):
    records = [episode(tmp_path, "a", source="pseudo", confidence=[0.9, 0.2]*4),
               episode(tmp_path, "b", source="unlabeled")]
    dataset = MobileDataset(manifest(tmp_path, records), split="train", steps=2, buttons=2)
    assert len(dataset) == 4
    assert dataset[0]["weight"].tolist() == pytest.approx([0.9, 0.])


def test_manifest_rejects_escape(tmp_path):
    item = episode(tmp_path)
    item["path"] = "../outside.npz"
    with pytest.raises(ValueError):
        read_manifest(manifest(tmp_path, [item]))


def test_dataset_rejects_timestamp_reordering(tmp_path):
    item = episode(tmp_path)
    file = tmp_path / item["path"]
    with np.load(file) as stored:
        values = dict(stored)
    values["timestamps_ns"][3] = 0
    np.savez(file, **values)
    with pytest.raises(ValueError, match="Timestamps"):
        MobileDataset(manifest(tmp_path, [item]), split="train", steps=2, buttons=2)
