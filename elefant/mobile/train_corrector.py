"""Train a latest-frame correction model from recorded causal plan/correction pairs.

Train/validation recording IDs must be disjoint. This CLI does not record games,
generate expert labels, or run inputs on a phone.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .models import FastCorrector, HybridPrediction, hybrid_loss
from elefant.integration.checkpoints import write_json_atomic


def load_pairs(path, buttons):
    with np.load(path, allow_pickle=False) as file:
        data = dict(file)
    required = ("frames", "plan_controls", "plan_logits", "executed", "targets", "timing", "weight", "recording_ids")
    if any(k not in data for k in required):
        raise ValueError("Correction shard lacks required arrays")
    n = len(data["frames"])
    if not n or data["frames"].dtype != np.uint8 or data["frames"].ndim != 4 or data["frames"].shape[1] != 3:
        raise ValueError("Expected RGB uint8 frames [N,3,H,W]")
    shapes = {"plan_controls": (n,4), "plan_logits": (n,buttons), "executed": (n,4+buttons),
              "targets": (n,4+buttons), "timing": (n,2), "weight": (n,)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid correction array: {key}")
    if (data["timing"] < 0).any() or (np.abs(data["plan_controls"]) > 1).any():
        raise ValueError("Negative plan age or unnormalized controls")
    for key in ("executed", "targets"):
        if (np.abs(data[key][:,:4]) > 1).any() or ((data[key][:,4:] != 0)&(data[key][:,4:] != 1)).any():
            raise ValueError("Correction actions require normalized controls and binary states")
    if ((data["weight"] < 0)|(data["weight"] > 1)).any() or not (data["weight"] > 0).any():
        raise ValueError("Correction weights must be in [0,1] with some positive labels")
    ids = data.pop("recording_ids")
    if ids.shape != (n,) or ids.dtype.kind not in "US" or any(not str(v) for v in ids):
        raise ValueError("Explicit recording_ids are required")
    return {k: torch.from_numpy(data[k]).to(torch.uint8 if k == "frames" else torch.float32)
            for k in required if k != "recording_ids"}, set(map(str, ids))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="outputs/corrector")
    parser.add_argument("--buttons", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    train, train_ids = load_pairs(args.train, args.buttons)
    valid, valid_ids = load_pairs(args.validation, args.buttons)
    if train_ids & valid_ids:
        raise ValueError("Recording leakage between corrector train/validation")
    torch.manual_seed(args.seed)
    model = FastCorrector(args.buttons, args.feature_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    losses = []
    def batch_loss(data, indices):
        batch = {k: v[indices].to(args.device) for k, v in data.items()}
        plan = HybridPrediction(batch["plan_controls"], batch["plan_logits"])
        result = model(batch["frames"], plan, batch["executed"], batch["timing"])
        return hybrid_loss(result, batch["targets"], batch["weight"])
    for _ in range(args.steps):
        indices = torch.randint(len(train["frames"]), (args.batch_size,))
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(train, indices)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite correction loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    weighted, weight_sum = 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(valid["frames"]), args.batch_size):
            indices = torch.arange(start, min(start+args.batch_size, len(valid["frames"])))
            w = float(valid["weight"][indices].sum())
            weighted += float(batch_loss(valid, indices))*max(w, 1.0)
            weight_sum += w
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "format": "mobile-corrector-v1",
                "buttons": args.buttons, "feature_dim": args.feature_dim, "steps": args.steps}, out / "corrector.ckpt")
    write_json_atomic(out / "training.json", {"steps": args.steps, "losses": losses,
                      "validation_loss": weighted/max(weight_sum, 1.0), "device": args.device,
                      "recording_groups_disjoint": True, "gameplay_verified": False})


if __name__ == "__main__":
    main()
