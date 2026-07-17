#!/usr/bin/env python3
"""Reproducible local/remote MUSDB18-HQ training entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ternarystem.config import load_config, model_config
from ternarystem.data import MUSDBChunkDataset, validate_track_names
from ternarystem.evaluation import base_record, save_record
from ternarystem.losses import complex_l1, global_sdr, multiresolution_stft_loss
from ternarystem.models import Separator

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/experiment.yaml")
parser.add_argument("--data-root", type=Path, help="MUSDB18-HQ train directory")
parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
parser.add_argument("--resume", type=Path)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
config = load_config(args.config)
torch.manual_seed(config["seed"])
model_cfg = model_config(config)
model = Separator(model_cfg)
parameters = sum(parameter.numel() for parameter in model.parameters())
print(json.dumps({"parameters": parameters, "model_config": asdict(model_cfg)}, indent=2))
if args.dry_run:
    raise SystemExit(0)
if args.data_root is None:
    raise SystemExit("--data-root is required unless --dry-run is used")

track_names = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
train_names, validation_names = validate_track_names(track_names)
chunk_samples = round(config["data"]["sample_rate"] * config["data"]["chunk_seconds"])
dataset = MUSDBChunkDataset(
    args.data_root,
    train_names,
    chunk_samples,
    config["data"]["epoch_chunks"],
    config["seed"],
)
loader = DataLoader(
    dataset,
    batch_size=config["train"]["batch_size"],
    num_workers=args.workers,
    pin_memory=args.device.startswith("cuda"),
)
validation_dataset = MUSDBChunkDataset(
    args.data_root,
    validation_names,
    chunk_samples,
    config["data"].get("validation_chunks", 140),
    config["seed"] + 1,
    remix=False,
    augment=False,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=config["train"]["batch_size"],
    num_workers=args.workers,
    pin_memory=args.device.startswith("cuda"),
)
device = torch.device(args.device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
start_epoch = 0
if args.resume:
    payload = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    start_epoch = payload["epoch"] + 1
args.output_dir.mkdir(parents=True, exist_ok=True)
record = base_record(config, config["seed"])
record["training"] = []
weights = config["train"]

for epoch in range(start_epoch, config["train"]["epochs"]):
    dataset.set_epoch(epoch)
    model.train()
    sums = {"loss": 0.0, "waveform_l1": 0.0, "complex_l1": 0.0, "sdr": 0.0}
    batches = 0
    for mixture, targets in loader:
        mixture, targets = mixture.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        estimate_spectra = model.spectrograms(mixture)
        estimates = model.stft.synthesis(estimate_spectra, mixture.shape[-1])
        target_spectra = model.stft.analysis(targets)
        waveform_loss = F.l1_loss(estimates, targets)
        spectrum_loss = complex_l1(estimate_spectra, target_spectra)
        loss = weights["waveform_l1"] * waveform_loss + weights["complex_l1"] * spectrum_loss
        if weights.get("multires_stft", 0) > 0:
            loss = loss + weights["multires_stft"] * multiresolution_stft_loss(
                estimates, targets
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        sums["loss"] += float(loss.detach())
        sums["waveform_l1"] += float(waveform_loss.detach())
        sums["complex_l1"] += float(spectrum_loss.detach())
        sums["sdr"] += float(global_sdr(estimates.detach(), targets))
        batches += 1
    metrics = {key: value / max(1, batches) for key, value in sums.items()}
    metrics["epoch"] = epoch
    model.eval()
    validation_sdr = validation_l1 = 0.0
    validation_batches = 0
    with torch.inference_mode():
        for mixture, targets in validation_loader:
            mixture, targets = mixture.to(device), targets.to(device)
            estimates = model(mixture)
            validation_l1 += float(F.l1_loss(estimates, targets))
            validation_sdr += float(global_sdr(estimates, targets))
            validation_batches += 1
    metrics["validation_waveform_l1"] = validation_l1 / max(1, validation_batches)
    metrics["validation_global_sdr"] = validation_sdr / max(1, validation_batches)
    record["training"].append(metrics)
    checkpoint = {
        "config": asdict(model_cfg),
        "resolved_config": config,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, args.output_dir / "latest.pt")
    previous = record["training"][:-1]
    if not previous or metrics["validation_global_sdr"] > max(
        item["validation_global_sdr"] for item in previous
    ):
        torch.save(checkpoint, args.output_dir / "best.pt")
    save_record(args.output_dir / "experiment.json", record)
    print(json.dumps(metrics))
