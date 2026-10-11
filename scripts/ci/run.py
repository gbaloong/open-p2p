"""One component-CI entry for local Linux/WSL, self-hosted and hosted Actions.

Uses the CURRENT interpreter (normally the bootstrap venv). No installation,
checkout cleaning, hardware training, game control or model download occurs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def junit_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    result = {"tests": len(cases), "failed": 0, "errors": 0, "skipped": 0}
    for case in cases:
        result["failed"] += case.find("failure") is not None
        result["errors"] += case.find("error") is not None
        result["skipped"] += case.find("skipped") is not None
    if not cases or any(result[key] for key in ("failed", "errors", "skipped")):
        raise ValueError(f"Executed-test gate failed: {result}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/"local-ci")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds < 1:
        parser.error("Timeout must be positive")
    # A fresh report directory prevents old green XML being reused after a failure.
    out = args.output.expanduser().resolve()/f"run-{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(ROOT),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", WANDB_MODE="disabled",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Do not automatically touch a GPU during baseline tests on a shared server.
    if args.profile == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    probe = [sys.executable, "scripts/ci/probe.py", "--output", str(out/"environment.json")]
    if args.profile == "cuda":
        probe.append("--require-cuda")
    commands = [
        ("environment", probe),
        ("tests", [sys.executable, "-m", "pytest", "-q", "tests/integration", "tests/ci",
                   f"--junitxml={out/'tests.xml'}"]),
        ("syntax", [sys.executable, "-m", "compileall", "-q", "elefant/integration", "elefant/mobile",
                    "elefant/policy_model/train.py", "scripts/ci"]),
        ("download_plan", [sys.executable, "scripts/download_checkpoints.py", "1200M", "--dry-run"]),
    ]
    report = {"scope": "component_ci_only", "profile": args.profile,
              "started_utc": datetime.now(timezone.utc).isoformat(), "steps": [], "passed": False,
              "model_downloaded": False, "full_model_training_verified": False}
    exit_code = 1
    try:
        for name, command in commands:
            print(f"== {name} ==", flush=True)
            start = time.perf_counter()
            with (out/f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout_seconds, check=False)
            print((out/f"{name}.log").read_text(encoding="utf-8"), flush=True)
            report["steps"].append({"name": name, "returncode": completed.returncode,
                                    "elapsed_seconds": time.perf_counter()-start})
            if completed.returncode != 0:
                raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
            if name == "tests":
                report["test_counts"] = junit_counts(out/"tests.xml")
        report["passed"] = True
        exit_code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(f"CI report: {out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
