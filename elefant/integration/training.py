"""Explicit single-GPU transfer/resume runner for the ORIGINAL PC policy.

DeepSpeed and the full upstream training dependencies are imported only at launch.
Mobile-policy training has its own runner; a PC checkpoint is never mislabeled as mobile.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import time

from .checkpoints import initialize_weights, write_json_atomic


def add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", help="Load only pretrained model tensors.")
    source.add_argument("--resume", help="Restore our complete Lightning/ZeRO training checkpoint.")
    source.add_argument("--from-scratch", action="store_true", help="Explicit random initialization.")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow trusted pickle loading.")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--checkpoint-ffn", action="store_true",
                        help="Non-reentrant activation checkpointing of feed-forward blocks.")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", default="outputs/transfer")


def validate_launch(args) -> None:
    sources = bool(args.init_checkpoint) + bool(args.resume) + bool(args.from_scratch)
    if sources != 1:
        raise ValueError("Choose exactly one initialization or resume source.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This first offload/accumulation runner is single-process only.")
    if args.accumulate < 1:
        raise ValueError("--accumulate must be positive.")
    if args.sequence_length is not None and args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive.")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive.")
    if args.resume and not Path(args.resume).exists():
        raise ValueError("Resume checkpoint does not exist.")


def prepare_config(config, args):
    config = config.model_copy(deep=True)
    config.shared.output_path = str(Path(args.output).resolve())
    if args.sequence_length is not None:
        config.shared.n_seq_timesteps = args.sequence_length
        config.policy_model.attention_history_len = [
            min(v, args.sequence_length) for v in config.policy_model.attention_history_len
        ]
        config.policy_model.attention_history_len[-1] = args.sequence_length
    if args.max_steps is not None:
        config.stage3_finetune.n_training_steps = args.max_steps
    # The upstream queue adapter asserts accumulation==1. It is a data producer,
    # not an optimizer; actual accumulation is configured on Trainer below.
    # Never apply this workaround to multi-rank upstream queues.
    config.stage3_finetune.accumulate_grad_batches = 1
    config.stage3_finetune.freeze_transformer_layers_for_steps = 0
    return config


def checkpoint_feedforwards(model) -> int:
    from functools import partial
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
    )
    from elefant.policy_model.transformer import PackedSwiGLUFFN
    count = sum(isinstance(m, PackedSwiGLUFFN) for m in model.modules())
    if not count:
        raise ValueError("No supported FFN blocks found; checkpointing was not applied.")
    # Do NOT wrap TransformerSelfAttentionLayer: upstream uses isinstance on it
    # for attention masks and KV-cache routing. Wrapping only FFNs preserves that.
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, PackedSwiGLUFFN),
    )
    return count


def make_strategy(cpu_offload: bool):
    if not cpu_offload:
        return "auto"
    from lightning.pytorch.strategies import DeepSpeedStrategy
    try:
        import deepspeed  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install DeepSpeed in the project environment before CPU offload.") from exc
    return DeepSpeedStrategy(stage=2, offload_optimizer=True, pin_memory=True,
                             allgather_bucket_size=5_000_000,
                             reduce_bucket_size=5_000_000, overlap_comm=False)


def make_optimizer(parameters, config, cpu_offload: bool):
    import torch
    optim = config.stage3_finetune.optim
    kwargs = dict(lr=optim.learning_rate, betas=(optim.beta_1, optim.beta_2),
                  weight_decay=optim.weight_decay)
    if cpu_offload:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(parameters, **kwargs)
    return torch.optim.AdamW(parameters, **kwargs)


def run_transfer(config, args) -> None:
    validate_launch(args)
    import torch
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, Callback
    from elefant.policy_model.stage3_finetune import Stage3LabelledBCLightning, Stage3DataModule
    if not torch.cuda.is_available():
        raise RuntimeError("Full upstream policy training requires a supported CUDA GPU.")
    config = prepare_config(config, args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Metadata is written even if subsequent GPU validation fails.
    write_json_atomic(out / "launch.json", {"kind": "pc_policy_transfer", "args": vars(args),
                                          "config": config.model_dump(mode="json"),
                                          "gpu_verified": False})

    class TransferPolicy(Stage3LabelledBCLightning):
        def configure_optimizers(self):
            return make_optimizer(self.parameters(), config, args.cpu_offload)

    class StepTimer(Callback):
        def __init__(self):
            self.times = []
            self.frames = 0
        def on_train_start(self, trainer, pl_module):
            torch.cuda.reset_peak_memory_stats()
            self.wall_started = time.perf_counter()
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            torch.cuda.synchronize()
            self.started = time.perf_counter()
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            self.times.append(time.perf_counter() - self.started)
            self.frames += int(batch.frames.shape[0]*batch.frames.shape[1])
        def on_train_end(self, trainer, pl_module):
            write_json_atomic(out / "measured_training.json", {
                "microbatch_seconds_including_update_when_due": self.times,
                "microbatch_timer_excludes_data_wait": True,
                "training_wall_seconds": time.perf_counter()-self.wall_started,
                "processed_frames": self.frames,
                "frames_per_second_including_data_wait": self.frames/max(time.perf_counter()-self.wall_started, 1e-9),
                "optimizer_steps": trainer.global_step,
                "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu": torch.cuda.get_device_name(), "accumulation": args.accumulate,
                "not_a_gameplay_evaluation": True})

    model = TransferPolicy(config)
    if args.init_checkpoint:
        report = initialize_weights(model, args.init_checkpoint, trust_pickle=args.trust_checkpoint)
        write_json_atomic(out / "initialization.json", report)
    if args.no_compile or args.cpu_offload or args.checkpoint_ffn:
        # Keep custom upstream flex-attention kernels; avoid compiling the whole
        # Lightning training-step around CPU-offload/checkpoint machinery.
        model.compile_mode = lambda fn: fn
    if args.checkpoint_ffn:
        checkpoint_feedforwards(model)
    model.requires_grad_(True)
    checkpoint = ModelCheckpoint(dirpath=out / "checkpoints", save_last=True,
                                  every_n_train_steps=config.stage3_finetune.save_every_n_steps,
                                  save_top_k=1)
    trainer = pl.Trainer(accelerator="gpu", devices=1, strategy=make_strategy(args.cpu_offload),
                         precision=config.shared.precision, max_steps=config.stage3_finetune.n_training_steps,
                         accumulate_grad_batches=args.accumulate, logger=False,
                         callbacks=[checkpoint, StepTimer()], num_sanity_val_steps=0,
                         limit_val_batches=0, fast_dev_run=config.shared.fast_dev_run)
    trainer.fit(model, datamodule=Stage3DataModule(config), ckpt_path=args.resume)
    # Under DeepSpeed this path is a DIRECTORY of shards. Use --resume on it;
    # consolidate before using it as a new --init-checkpoint weights file.
    trainer.save_checkpoint(out / "final.ckpt")
