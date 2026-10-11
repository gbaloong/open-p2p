"""Report bounded CI environment facts and execute a tiny optimizer update.

CUDA success here is NOT a 1.2B/DeepSpeed or Android benchmark.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def inspect_environment(require_cuda: bool = False) -> dict:
    import torch
    device = "cuda" if require_cuda else "cpu"
    report = {"scope": "component_ci_only", "python": platform.python_version(),
              "os": platform.system(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
              "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
              "profile": device, "gpu_probe_requested": require_cuda,
              "deepspeed_installed": importlib.util.find_spec("deepspeed") is not None,
              "full_project_environment": False, "model_1200m_loaded": False,
              "android_tested": False, "optimizer_update_passed": False}
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU fallback would invalidate this check")
        report["gpu"] = torch.cuda.get_device_name()
        report["capability"] = list(torch.cuda.get_device_capability())
        report["compiled_architectures"] = torch.cuda.get_arch_list()
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        if not report["bf16_supported"]:
            raise RuntimeError("This CUDA smoke profile requires native BF16 support")
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.weight.detach().clone()
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=require_cuda)
    with context:
        loss = model(torch.ones(4, 16, device=device)).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(model.weight.grad).all():
        raise RuntimeError("Non-finite tiny-model loss or gradient")
    optimizer.step()
    if require_cuda:
        torch.cuda.synchronize()
    if torch.equal(before, model.weight):
        raise RuntimeError("Optimizer did not update the tiny model")
    report["optimizer_update_passed"] = True
    try:
        report["code_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        report["code_revision"] = None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = inspect_environment(args.require_cuda)
        status = 0
    except Exception as exc:
        report = {"scope": "component_ci_only", "passed": False,
                  "error": f"{type(exc).__name__}: {exc}", "model_1200m_loaded": False}
        status = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
