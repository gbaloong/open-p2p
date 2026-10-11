"""Train the small OFFLINE latent-action baseline on episode-grouped visual features."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .video_learning import LatentActionVQ
from elefant.integration.checkpoints import write_json_atomic


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="NPZ: features[N,D], episode_ids[N]")
    parser.add_argument("--output", default="outputs/video-latents")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    with np.load(args.features, allow_pickle=False) as archive:
        features, episodes = archive["features"], archive["episode_ids"]
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("Invalid features")
    if episodes.shape != (len(features),) or episodes.dtype.kind not in "USiu":
        raise ValueError("Explicit episode IDs are required")
    pairs = np.where(episodes[:-1] == episodes[1:])[0]
    if not len(pairs):
        raise ValueError("No within-episode adjacent frames")
    tensor = torch.from_numpy(features.astype(np.float32))
    model = LatentActionVQ(features.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(args.steps):
        idx = torch.from_numpy(pairs)[torch.randint(len(pairs), (args.batch_size,))]
        result = model(tensor[idx].to(args.device), tensor[idx+1].to(args.device))
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "feature_dim": features.shape[1],
                "format": "offline-latent-baseline-v1"}, out / "latents.ckpt")
    write_json_atomic(out / "training.json", {"losses": losses, "within_episode_pairs": len(pairs),
                      "not_touch_labels": True, "heldout_evaluation_performed": False})


if __name__ == "__main__":
    main()
