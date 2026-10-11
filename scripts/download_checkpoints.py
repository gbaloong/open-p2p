"""Download a selected model size without installing the full training stack."""
import argparse
from pathlib import Path
import json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=["150M", "300M", "600M", "1200M"])
    parser.add_argument("--repo-id", default="elefantai/open-p2p")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"repo_id": args.repo_id, "revision": args.revision,
                          "allow_patterns": f"{args.size}/*", "output": str(args.output)}))
        return
    from huggingface_hub import HfApi, snapshot_download
    # Resolve once, then pin the download to that exact revision.
    sha = HfApi().model_info(args.repo_id, revision=args.revision).sha
    snapshot_download(repo_id=args.repo_id, revision=sha,
                      allow_patterns=f"{args.size}/*", local_dir=str(args.output))
    expected = args.output / args.size / "checkpoint-step=00500000.ckpt"
    if not expected.is_file() or expected.stat().st_size == 0:
        raise RuntimeError(f"Download did not produce the expected checkpoint: {expected}")
    manifest = {"repo_id": args.repo_id, "revision": sha, "size": args.size,
                "checkpoint": expected.name, "bytes": expected.stat().st_size,
                "config_source": f"config/policy_model/{args.size}.yaml"}
    (expected.parent / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {expected}. Use repository config, not bundled model_config.yaml.")


if __name__ == "__main__":
    main()
