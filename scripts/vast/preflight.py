#!/usr/bin/env python3
"""Fail-closed Vast/cloud GPU, MUSDB18-HQ, and training-shape preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from ternarystem.config import load_config, model_config
from ternarystem.data import STEMS, validate_track_names
from ternarystem.models import Separator
from ternarystem.training import atomic_json_save


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fail(message: str) -> None:
    raise SystemExit(f"PREFLIGHT FAILED: {message}")


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--min-free-gb", type=float, default=20.0)
parser.add_argument("--expected-commit")
parser.add_argument("--allow-dirty", action="store_true")
parser.add_argument(
    "--accept-musdb-terms",
    action="store_true",
    help="assert that the operator supplied MUSDB18-HQ under acceptable terms",
)
args = parser.parse_args()

if not args.accept_musdb_terms:
    fail("pass --accept-musdb-terms after legally obtaining and supplying MUSDB18-HQ")
if not args.config.is_file():
    fail(f"config does not exist: {args.config}")
if not args.data_root.is_dir():
    fail(f"dataset directory does not exist: {args.data_root}")
if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is false; never pay for a silent CPU fallback")
if torch.version.cuda is None:
    fail("the installed PyTorch build has no CUDA runtime")

commit = git("rev-parse", "HEAD")
dirty_output = git("status", "--porcelain")
if args.expected_commit and commit != args.expected_commit:
    fail(f"expected source commit {args.expected_commit}, found {commit}")
if dirty_output and not args.allow_dirty:
    fail("source tree is dirty; commit the exact training source or pass --allow-dirty")

config = load_config(args.config)
track_names = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
try:
    train_names, validation_names = validate_track_names(track_names)
except (ValueError, FileNotFoundError) as error:
    fail(str(error))

manifest = []
for name in track_names:
    frame_counts = set()
    for stem in STEMS:
        path = args.data_root / name / f"{stem}.wav"
        if not path.is_file():
            fail(f"missing stem: {path}")
        try:
            info = sf.info(path)
        except RuntimeError as error:
            fail(f"cannot decode metadata for {path}: {error}")
        if info.samplerate != 44100 or info.channels != 2 or info.frames <= 0:
            fail(f"expected non-empty stereo 44.1 kHz audio: {path}")
        frame_counts.add(info.frames)
        # Decode both ends so common truncation/corruption failures are caught before rent burns.
        try:
            sf.read(path, start=0, frames=min(4096, info.frames), dtype="float32")
            sf.read(
                path,
                start=max(0, info.frames - 4096),
                frames=min(4096, info.frames),
                dtype="float32",
            )
        except RuntimeError as error:
            fail(f"cannot decode {path}: {error}")
        manifest.append(
            {
                "track": name,
                "stem": stem,
                "frames": info.frames,
                "bytes": path.stat().st_size,
                "format": info.format,
                "subtype": info.subtype,
            }
        )
    if len(frame_counts) != 1:
        fail(f"stem lengths disagree for track: {name}")

usage = shutil.disk_usage(args.output.parent if args.output.parent.exists() else Path.cwd())
free_gb = usage.free / 1024**3
if free_gb < args.min_free_gb:
    fail(f"only {free_gb:.1f} GiB free; require {args.min_free_gb:.1f} GiB")

try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
except (OSError, subprocess.CalledProcessError) as error:
    fail(f"nvidia-smi failed: {error}")

# Exercise the actual configured batch/chunk shape, including backward and optimizer state.
resolved = model_config(config)
device = torch.device("cuda")
torch.manual_seed(int(config["seed"]))
model = Separator(resolved).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]))
batch_size = int(config["train"]["batch_size"])
samples = round(float(config["data"]["sample_rate"]) * float(config["data"]["chunk_seconds"]))
step_started = time.perf_counter()
try:
    mixture = torch.randn(batch_size, 2, samples, device=device) * 0.01
    target = mixture[:, None].expand(-1, resolved.sources, -1, -1) / resolved.sources
    estimate = model(mixture)
    loss = F.l1_loss(estimate, target)
    if not torch.isfinite(loss):
        fail("representative CUDA step produced a non-finite loss")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
except torch.cuda.OutOfMemoryError as error:
    fail(f"representative configured batch/chunk does not fit GPU memory: {error}")
except RuntimeError as error:
    fail(f"representative CUDA training step failed: {error}")
step_seconds = time.perf_counter() - step_started

manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
report = {
    "status": "passed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source": {"commit": commit, "dirty": bool(dirty_output)},
    "config": str(args.config.resolve()),
    "data": {
        "root": str(args.data_root.resolve()),
        "license_asserted_by_operator": True,
        "tracks": len(track_names),
        "train_tracks": len(train_names),
        "validation_tracks": len(validation_names),
        "stem_files": len(manifest),
        "metadata_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    },
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": smi,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "free_disk_gb": free_gb,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    },
    "representative_step": {
        "batch_size": batch_size,
        "samples": samples,
        "loss": float(loss.detach()),
        "cold_step_seconds": step_seconds,
        "passed": True,
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.is_file():
    previous = json.loads(args.output.read_text(encoding="utf-8"))
    previous_hash = previous.get("data", {}).get("metadata_manifest_sha256")
    current_hash = report["data"]["metadata_manifest_sha256"]
    if previous_hash != current_hash:
        fail("dataset metadata manifest changed since this run was created")
    report["first_preflight_utc"] = previous.get(
        "first_preflight_utc", previous.get("created_utc")
    )
else:
    report["first_preflight_utc"] = report["created_utc"]
atomic_json_save(report, args.output)
print(json.dumps(report, indent=2))
