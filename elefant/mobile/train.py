"""Train the mobile head/shared core on timestamped NPZ episodes; no device control."""
from __future__ import annotations
import argparse
from pathlib import Path
import time

from elefant.integration.training import (add_transfer_arguments, validate_launch, prepare_config,
                                          make_optimizer, make_strategy, checkpoint_feedforwards)
from elefant.integration.checkpoints import initialize_weights, write_json_atomic


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/policy_model/1200M.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--buttons", type=int, default=8)
    parser.add_argument("--goal-dim", type=int, default=32)
    parser.add_argument("--delay-frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    add_transfer_arguments(parser)
    args = parser.parse_args(argv)
    validate_launch(args)
    if args.workers < 0 or not 1 <= args.buttons <= 8 or args.goal_dim < 1:
        parser.error("Invalid workers, button count or goal dimension")
    import torch
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from elefant.config import load_config
    from elefant.policy_model.config import LightningPolicyConfig
    from elefant.policy_model.stage3_finetune import Stage3LabelledBCLightning
    from elefant.torch import pytorch_setup
    from .policy import MobilePolicy
    from .data import MobileDataset
    from .models import hybrid_loss
    if not torch.cuda.is_available():
        raise RuntimeError("The upstream full model requires CUDA; use CPU unit tests for modules")
    pytorch_setup()
    config = prepare_config(load_config(args.config, LightningPolicyConfig), args)
    steps = config.shared.n_seq_timesteps
    train = MobileDataset(args.manifest, split="train", steps=steps, buttons=args.buttons,
                          goal_dim=args.goal_dim, delay_frames=args.delay_frames)
    valid = MobileDataset(args.manifest, split="validation", steps=steps, buttons=args.buttons,
                          goal_dim=args.goal_dim, delay_frames=args.delay_frames)
    base = Stage3LabelledBCLightning(config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.init_checkpoint:
        report = initialize_weights(base, args.init_checkpoint, trust_pickle=args.trust_checkpoint)
        write_json_atomic(out / "initialization.json", report)
    policy = MobilePolicy.from_pretrained_policy(base, buttons=args.buttons, goal_dim=args.goal_dim)
    del base  # Release unused PC heads/metrics; retain the shared pretrained core.
    if args.checkpoint_ffn:
        checkpoint_feedforwards(policy.core._transformer)
    write_json_atomic(out / "mobile_launch.json", {"args": vars(args),
                      "config": config.model_dump(mode="json"), "parameters": policy.parameter_report(),
                      "gameplay_verified": False})

    class Learner(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.policy = policy
        def training_step(self, batch, batch_idx):
            pred = self.policy(batch["frames"], batch["executed"], batch["timing"],
                               batch["goals"], batch["visible"])
            loss = hybrid_loss(pred, batch["targets"], batch["weight"])
            self.log("train_loss", loss, batch_size=len(batch["frames"]))
            return loss
        def validation_step(self, batch, batch_idx):
            pred = self.policy(batch["frames"], batch["executed"], batch["timing"],
                               batch["goals"], batch["visible"])
            loss = hybrid_loss(pred, batch["targets"], batch["weight"])
            self.log("validation_loss", loss, batch_size=len(batch["frames"]))
            return loss
        def configure_optimizers(self):
            return make_optimizer((p for p in self.parameters() if p.requires_grad), config, args.cpu_offload)

    loader_args = dict(batch_size=1, num_workers=args.workers, pin_memory=True)
    callback = ModelCheckpoint(dirpath=out / "checkpoints", save_last=True, save_top_k=1,
                                monitor="validation_loss", mode="min")
    trainer = pl.Trainer(accelerator="gpu", devices=1, strategy=make_strategy(args.cpu_offload),
                         precision=config.shared.precision, max_steps=config.stage3_finetune.n_training_steps,
                         accumulate_grad_batches=args.accumulate, logger=False, callbacks=[callback],
                         num_sanity_val_steps=0, fast_dev_run=args.fast_dev_run)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.fit(Learner(), DataLoader(train, shuffle=True, **loader_args),
                DataLoader(valid, shuffle=False, **loader_args), ckpt_path=args.resume)
    torch.cuda.synchronize()
    trainer.save_checkpoint(out / "final.ckpt")
    write_json_atomic(out / "measured_training.json", {"elapsed_seconds": time.perf_counter()-start,
                      "optimizer_steps": trainer.global_step, "gpu": torch.cuda.get_device_name(),
                      "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "not_a_gameplay_evaluation": True})


if __name__ == "__main__":
    main()
