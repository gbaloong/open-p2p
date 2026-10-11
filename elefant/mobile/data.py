"""Episode-level JSONL/NPZ dataset with causal delayed observations and label provenance.

NPZ arrays: frames [N,3,H,W] uint8, executed [N,4+B], targets [N,4+B],
timestamps_ns [N] int64, confidence [N] optional, goals/visible optional.
Each actions[t] is the action AFTER frame[t]; only t'<t is visible to output[t].
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset


SOURCES = {"logged", "corrected", "pseudo", "unlabeled"}


def read_manifest(path: str | Path) -> tuple[Path, list[dict]]:
    path = Path(path).resolve()
    records, group_splits, seen_files, ids = [], {}, set(), set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        for key in ("episode_id", "recording_id", "split", "path", "label_source"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Line {line_no}: missing {key}")
        if record["split"] not in {"train", "validation", "test"} or record["label_source"] not in SOURCES:
            raise ValueError(f"Line {line_no}: invalid split/source")
        if record["episode_id"] in ids:
            raise ValueError("Duplicate episode id")
        ids.add(record["episode_id"])
        group = record["recording_id"]
        if group in group_splits and group_splits[group] != record["split"]:
            raise ValueError("Train/test leakage: one recording appears in different splits")
        group_splits[group] = record["split"]
        full = (path.parent / record["path"]).resolve()
        if not full.is_relative_to(path.parent) or full.suffix != ".npz" or not full.is_file():
            raise ValueError("Episode path must reference an existing .npz inside the manifest root")
        if full in seen_files:
            raise ValueError("Same episode file listed more than once")
        seen_files.add(full)
        record = {**record, "resolved_path": str(full)}
        records.append(record)
    if not records:
        raise ValueError("Empty dataset manifest")
    return path.parent, records


def load_episode(record, buttons: int, goal_dim: int):
    with np.load(record["resolved_path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    for key in ("frames", "executed", "targets", "timestamps_ns"):
        if key not in data:
            raise ValueError(f"Missing episode array: {key}")
    frames = data["frames"]
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1] != 3 or len(frames) == 0:
        raise ValueError("Invalid RGB frame array")
    n = len(frames)
    ts = data["timestamps_ns"]
    if ts.shape != (n,) or not np.issubdtype(ts.dtype, np.integer) or (ts < 0).any() or (np.diff(ts) <= 0).any():
        raise ValueError("Timestamps must be strictly increasing nonnegative integers")
    for key in ("executed", "targets"):
        value = data[key]
        if value.shape != (n, 4+buttons) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} actions")
        if (np.abs(value[:, :4]) > 1).any() or ((value[:, 4:] != 0) & (value[:, 4:] != 1)).any():
            raise ValueError(f"{key} must have normalized controls and binary button states")
        data[key] = value.astype(np.float32)
    if record["label_source"] == "pseudo" and "confidence" not in data:
        raise ValueError("Pseudo labels require explicit confidence")
    confidence = data.get("confidence", np.ones(n, dtype=np.float32))
    if confidence.shape != (n,) or not np.isfinite(confidence).all() or ((confidence < 0)|(confidence > 1)).any():
        raise ValueError("Invalid label confidence")
    if record["label_source"] == "unlabeled":
        confidence = np.zeros(n, dtype=np.float32)
    data["confidence"] = confidence.astype(np.float32)
    goals = data.get("goals", np.zeros((n, goal_dim), dtype=np.float32))
    visible = data.get("visible", np.zeros(n, dtype=bool))
    if goals.shape != (n, goal_dim) or not np.isfinite(goals).all() or visible.shape != (n,) or visible.dtype != bool:
        raise ValueError("Invalid goal arrays; visible must be bool")
    data["goals"], data["visible"] = goals.astype(np.float32), visible
    return data


class MobileDataset(Dataset):
    """Non-overlapping windows; one cached episode per worker, never a whole-video RAM cache."""
    def __init__(self, manifest, *, split, steps, buttons=8, goal_dim=32,
                 delay_frames=0, confidence_threshold=0.8):
        if steps < 1 or delay_frames < 0 or not 0 <= confidence_threshold <= 1:
            raise ValueError("Invalid dataset window/delay/confidence")
        _, records = read_manifest(manifest)  # Validate ALL splits before filtering.
        self.records = [r for r in records if r["split"] == split and r["label_source"] != "unlabeled"]
        self.steps, self.buttons, self.goal_dim = steps, buttons, goal_dim
        self.delay, self.threshold = delay_frames, confidence_threshold
        self.index = []
        self._cached_idx, self._cached = None, None
        for i, record in enumerate(self.records):
            data = load_episode(record, buttons, goal_dim)
            for start in range(delay_frames, len(data["frames"])-steps+1, steps):
                w = data["confidence"][start:start+steps]
                valid = w >= confidence_threshold if record["label_source"] == "pseudo" else w > 0
                if valid.any():
                    self.index.append((i, start))
        if not self.index:
            raise ValueError(f"No labelled windows in {split}; no pseudo labels were fabricated")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode_index, start = self.index[index]
        record = self.records[episode_index]
        if episode_index != self._cached_idx:
            self._cached = load_episode(record, self.buttons, self.goal_dim)
            self._cached_idx = episode_index
        data = self._cached
        stop = start+self.steps
        obs = slice(start-self.delay, stop-self.delay)
        current = slice(start, stop)
        timestamps = data["timestamps_ns"]
        age = (timestamps[current]-timestamps[obs]).astype(np.float64)/1e9
        prev = np.maximum(np.arange(start, stop)-1, 0)
        delta = (timestamps[current]-timestamps[prev]).astype(np.float64)/1e9
        confidence = data["confidence"][current].copy()
        if record["label_source"] == "pseudo":
            confidence[confidence < self.threshold] = 0
        # Goal visibility belongs to OBSERVED time, not the future target time.
        return {"frames": torch.from_numpy(data["frames"][obs].copy()),
                "executed": torch.from_numpy(data["executed"][current].copy()),
                "targets": torch.from_numpy(data["targets"][current].copy()),
                "weight": torch.from_numpy(confidence),
                "timing": torch.from_numpy(np.stack((age, delta), -1).astype(np.float32)),
                "goals": torch.from_numpy(data["goals"][obs].copy()),
                "visible": torch.from_numpy(data["visible"][obs].copy())}
