"""Create an isolated, version-keyed CI venv without changing training Python.

Linux/WSL. No sudo, runner registration, model downloads or project-wide uv sync.
The lock prevents concurrent jobs from installing into the same cached venv.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TORCH_VERSION = "2.10.0"
INDEXES = {"cpu": "https://download.pytorch.org/whl/cpu",
           "cuda": "https://download.pytorch.org/whl/cu128"}
REQUIREMENTS = Path(__file__).with_name("requirements.txt")


def environment_spec(profile: str) -> dict:
    if profile not in INDEXES:
        raise ValueError("Profile must be cpu or cuda")
    return {"schema": 1, "profile": profile, "torch": TORCH_VERSION,
            "index": INDEXES[profile], "python": sys.version.split()[0],
            "requirements": REQUIREMENTS.read_text(encoding="utf-8")}


def environment_path(profile: str, cache: Path) -> Path:
    fingerprint = hashlib.sha256(json.dumps(environment_spec(profile), sort_keys=True).encode()).hexdigest()[:20]
    return cache.expanduser().resolve() / f"{profile}-{fingerprint}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=INDEXES, default="cpu")
    parser.add_argument("--cache-root", type=Path,
                        default=Path(os.environ.get("OPENP2P_CI_CACHE", str(Path.home()/".cache"/"open-p2p-ci"))))
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error("Component CI supports Python 3.11-3.13; Actions uses 3.11. Full training is separate.")
    spec = environment_spec(args.profile)
    env_path = environment_path(args.profile, args.cache_root)
    python = env_path / "bin" / "python"
    marker = env_path / "open-p2p-ci.json"
    plan = {"profile": args.profile, "venv": str(env_path), "python": str(python),
            "torch_index": spec["index"], "model_download": False,
            "training_environment_modified": False}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if sys.platform != "linux":
        parser.error("Use Linux/WSL for this self-hosted setup; no host OS changes are performed")
    import fcntl
    env_path.parent.mkdir(parents=True, exist_ok=True)
    with (env_path.parent / f"{env_path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if env_path.is_symlink():
            raise RuntimeError("Refusing a symlink as the CI venv")
        if env_path.exists() and not (env_path/"pyvenv.cfg").is_file():
            raise RuntimeError("Existing directory is not a venv; refusing to overwrite it")
        if marker.is_file() and json.loads(marker.read_text()) != spec:
            raise RuntimeError("CI environment manifest mismatch; use a different cache root")
        if not python.exists():
            try:
                subprocess.run([sys.executable, "-m", "venv", str(env_path)], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("venv unavailable: provision python3-venv for the runner account; no sudo attempted") from exc
        env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        if not marker.is_file():
            subprocess.run([str(python), "-m", "pip", "install", "--index-url", spec["index"],
                            f"torch=={TORCH_VERSION}"], check=True, env=env)
            subprocess.run([str(python), "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                            "-r", str(REQUIREMENTS)], check=True, env=env)
        # Missing/corrupt dependencies fail rather than silently claiming a ready cache.
        subprocess.run([str(python), "-m", "pip", "check"], check=True, env=env)
        verify = ("import torch,numpy,pytest,yaml; "
                  "assert torch.__version__.split('+')[0]=='2.10.0'; "
                  "assert numpy.__version__=='2.3.5'; assert pytest.__version__=='9.0.2'; "
                  "assert yaml.__version__=='6.0.3'; "
                  + ("assert torch.version.cuda is None" if args.profile == "cpu"
                     else "assert torch.version.cuda=='12.8'"))
        subprocess.run([str(python), "-c", verify], check=True, env=env)
        temp = marker.with_suffix(".tmp")
        temp.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")
        temp.replace(marker)
    if args.github_output:
        if "\n" in str(python) or "\r" in str(python):
            raise ValueError("Invalid Python output path")
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"python={python}\n")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
