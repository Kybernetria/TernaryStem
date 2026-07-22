#!/usr/bin/env python3
"""Re-evaluate a checkpoint on deterministic development chunks without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ternarystem.data import MUSDBChunkDataset, validate_track_names
from ternarystem.evaluation import DevelopmentDiagnostics, base_record, save_record, sha256
from ternarystem.models import Separator, SeparatorConfig

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--chunks", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output", type=Path)
args = parser.parse_args()

payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
raw_config = dict(payload["config"])
raw_config["channels"] = tuple(raw_config["channels"])
model_config = SeparatorConfig(**raw_config)
resolved = payload.get("resolved_config") or {}
data_config = resolved.get("data") or {}
train_config = resolved.get("train") or {}
seed = int(resolved.get("seed", 20250218))
chunks = args.chunks or int(data_config.get("validation_chunks", 140))
batch_size = args.batch_size or int(train_config.get("batch_size", 1))
workers = args.workers if args.workers is not None else int(payload.get("data_workers", 2))
chunk_samples = round(
    float(data_config.get("sample_rate", 44100))
    * float(data_config.get("chunk_seconds", 6.0))
)
track_names = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
_, validation_names = validate_track_names(track_names)
dataset = MUSDBChunkDataset(
    args.data_root,
    validation_names,
    chunk_samples,
    chunks,
    seed + 1,
    remix=False,
    augment=False,
)
loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
device = torch.device(args.device)
model = Separator(model_config).to(device).eval()
model.load_state_dict(payload["state_dict"])
diagnostics = DevelopmentDiagnostics(model_config.sources)
with torch.inference_mode():
    for mixture, targets in loader:
        mixture, targets = mixture.to(device), targets.to(device)
        diagnostics.update(model(mixture), targets, mixture)
result = diagnostics.compute()
record = base_record(
    {
        "checkpoint": str(args.checkpoint),
        "chunks": chunks,
        "batch_size": batch_size,
        "workers": workers,
        "chunk_samples": chunk_samples,
        "validation_tracks": validation_names,
    },
    seed,
    args.device,
)
record["checkpoint_sha256"] = sha256(args.checkpoint)
record["quality"] = result
if args.output:
    save_record(args.output, record)
print(json.dumps(result, indent=2))
