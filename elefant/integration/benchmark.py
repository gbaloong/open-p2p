"""Measure the original pretrained PC policy on sequential frames without input injection."""
from __future__ import annotations
import argparse
import time
import numpy as np
from .checkpoints import initialize_weights, write_json_atomic


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Need a nonempty sequence of finite nonnegative durations")
    return {"count": len(values), "mean_ms": float(values.mean()*1000),
            "p50_ms": float(np.percentile(values, 50)*1000),
            "p95_ms": float(np.percentile(values, 95)*1000),
            "p99_ms": float(np.percentile(values, 99)*1000), "max_ms": float(values.max()*1000)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/policy_model/1200M.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", required=True, help="RGB uint8 NPY [N,3,192,192]")
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--output", default="outputs/baseline-latency.json")
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    import torch
    from elefant.config import load_config
    from elefant.policy_model.config import LightningPolicyConfig
    from elefant.policy_model.stage3_finetune import Stage3LabelledBCLightning
    from elefant.torch import pytorch_setup
    if not torch.cuda.is_available():
        raise RuntimeError("A supported NVIDIA GPU is needed to benchmark the upstream policy")
    config = load_config(args.config, LightningPolicyConfig)
    frames = np.load(args.frames, mmap_mode="r", allow_pickle=False)
    expected = (3, config.shared.frame_height, config.shared.frame_width)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != expected or len(frames) == 0:
        raise ValueError(f"Expected nonempty RGB uint8 [N,{expected}], got {frames.shape}")
    pytorch_setup()
    model = Stage3LabelledBCLightning(config, inference_mode=True)
    initialization = initialize_weights(model, args.checkpoint, trust_pickle=args.trust_checkpoint)
    model = model.eval().to("cuda")
    core = model.bc_transformer
    core.block_mask_to_device(torch.device("cuda"))
    core.setup_kv_cache(batch_size=1, device=torch.device("cuda"))
    warmup = max(args.warmup, config.shared.n_seq_timesteps+1)
    core.rebuild_rope_cache((warmup+args.iterations+2)*core.step_size)
    cache = core.init_kv_cache_state()
    idx = torch.zeros((), dtype=torch.int64, device="cuda")
    text = torch.zeros(core.text_token_size, core.text_tokenizer_embed_dim, device="cuda", dtype=torch.bfloat16)
    times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(warmup+args.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            frame = torch.from_numpy(np.array(frames[i % len(frames)], copy=True)).to("cuda")
            _, idx, cache = model.online_kv_cache_predict(frame=frame, idx=idx, kv_cache_state=cache,
                                                         text_tokens_embed=text, compile=args.compile)
            torch.cuda.synchronize()
            duration = time.perf_counter()-start
            if i >= warmup:
                times.append(duration)
    write_json_atomic(args.output, {"scope": "PC policy + frame copy; excludes phone capture and touch execution",
                      "initialization": initialization, "latency": latency_summary(times),
                      "warmup_steps": warmup, "history_steps": config.shared.n_seq_timesteps,
                      "compile": args.compile, "gpu": torch.cuda.get_device_name(),
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "raw_seconds": times, "is_mobile_gameplay_result": False})
    print(f"Measured report: {args.output}")


if __name__ == "__main__":
    main()
