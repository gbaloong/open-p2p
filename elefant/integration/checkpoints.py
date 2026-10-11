"""Fail-closed, weights-only initialization; resume is a separate Trainer operation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import os
import tempfile

import torch
from torch import nn


def read_weights(path: str | Path, *, trust_pickle: bool = False) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Expected a checkpoint file: {path}. Consolidate ZeRO shards first.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=not trust_pickle)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint loading failed. No unsafe fallback was attempted. "
            "Only use --trust-checkpoint with a verified, trusted pickle checkpoint."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must be a state dictionary or contain state_dict.")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Missing or empty state_dict.")
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("state_dict must contain only named tensors.")
        # torch.compile may insert this wrapper into nested module names.
        name = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if name in result:
            raise ValueError(f"Colliding checkpoint keys after canonicalization: {name}")
        result[name] = value
    return result


def initialize_weights(model: nn.Module, path: str | Path, *, trust_pickle: bool = False) -> dict:
    """Validate EVERY key and shape before mutating the model; do not load optimizer state."""
    state = read_weights(path, trust_pickle=trust_pickle)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = [
        (key, list(expected[key].shape), list(state[key].shape))
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(json.dumps({"missing": missing, "unexpected": unexpected,
                                     "shape_mismatches": shape_mismatches}, indent=2))
    model.load_state_dict(state, strict=True)
    return {"path": str(Path(path).resolve()), "loaded_tensors": len(state),
            "loaded_elements": sum(v.numel() for v in state.values()),
            "mode": "weights_only_initialization", "unsafe_pickle_enabled": trust_pickle}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)
