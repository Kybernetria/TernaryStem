#!/usr/bin/env python3
"""Reproducible local/remote MUSDB18-HQ training entry point."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ternarystem.config import load_config, model_config
from ternarystem.data import MUSDBChunkDataset, validate_track_names
from ternarystem.evaluation import DevelopmentDiagnostics, base_record, save_record, sha256
from ternarystem.losses import complex_l1, global_sdr, multiresolution_stft_loss
from ternarystem.models import Separator
from ternarystem.training import (
    atomic_torch_save,
    build_scheduler,
    build_teacher,
    distillation_config,
    load_checkpoint,
    prepare_teacher_targets,
    resume_training,
    ternary_training_summary,
    warm_start_model,
    waveform_distillation_l1,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/experiment.yaml")
parser.add_argument("--data-root", type=Path, help="MUSDB18-HQ train directory")
parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
checkpoint_group = parser.add_mutually_exclusive_group()
checkpoint_group.add_argument("--resume", type=Path, help="restore an interrupted run exactly")
checkpoint_group.add_argument(
    "--init-checkpoint", type=Path, help="warm-start model weights with a fresh optimizer"
)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument(
    "--require-cuda",
    action="store_true",
    help="fail instead of silently falling back to CPU on a rented GPU host",
)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
if args.require_cuda and (not torch.cuda.is_available() or not args.device.startswith("cuda")):
    raise SystemExit("CUDA was required, but the selected device is not an available CUDA device")
config = load_config(args.config)
distillation = distillation_config(config)
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
train_config = config["train"]
amp_mode = str(train_config.get("amp", "off")).lower()
if amp_mode not in {"off", "fp16", "bf16"}:
    raise ValueError("train.amp must be off, fp16, or bf16")
if amp_mode != "off" and device.type != "cuda":
    raise ValueError("train.amp requires a CUDA device")
amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
if hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")
else:  # PyTorch 2.2 compatibility
    scaler = torch.cuda.amp.GradScaler(enabled=amp_mode == "fp16")
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
scheduler = build_scheduler(optimizer, train_config)
start_epoch = 0
checkpoint_payload = None
missing_quantizer_keys: list[str] = []
if args.resume:
    checkpoint_payload = load_checkpoint(args.resume, device)
    if checkpoint_payload.get("resolved_config") != config:
        raise ValueError("exact resume requires the checkpoint's resolved configuration")
    start_epoch = resume_training(model, optimizer, checkpoint_payload, scheduler)
    if scaler.is_enabled():
        if not isinstance(checkpoint_payload.get("scaler"), dict):
            raise ValueError("FP16 resume checkpoint must contain gradient-scaler state")
        scaler.load_state_dict(checkpoint_payload["scaler"])
elif args.init_checkpoint:
    checkpoint_payload = load_checkpoint(args.init_checkpoint, device)
    missing_quantizer_keys = warm_start_model(model, checkpoint_payload)
if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
    raise SystemExit(
        f"refusing to start a new run in non-empty output directory: {args.output_dir}"
    )
teacher = build_teacher(distillation, device)
args.output_dir.mkdir(parents=True, exist_ok=True)
record_path = args.output_dir / "experiment.json"
latest_path = args.output_dir / "latest.pt"
best_path = args.output_dir / "best.pt"
if args.resume and record_path.is_file():
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.setdefault("resume_events", []).append(base_record(config, config["seed"], args.device))
else:
    record = base_record(config, config["seed"], args.device)
# A validated checkpoint is authoritative if a crash happened between checkpoint and
# experiment-record publication.
record["training"] = (
    list(checkpoint_payload.get("training_history", []))
    if args.resume and checkpoint_payload is not None
    else []
)
record["execution"] = {
    "argv": sys.argv,
    "data_root": str(args.data_root.resolve()),
    "workers": args.workers,
}
record["distillation"] = asdict(distillation)
record["mixed_precision_training"] = amp_mode
if not args.resume or "initialization" not in record:
    initialization_checkpoint = args.init_checkpoint if args.init_checkpoint else args.resume
    record["initialization"] = {
        "mode": "resume" if args.resume else "warm_start" if args.init_checkpoint else "scratch",
        "checkpoint": str(initialization_checkpoint) if initialization_checkpoint else None,
        "checkpoint_sha256": sha256(initialization_checkpoint)
        if initialization_checkpoint
        else None,
        "missing_quantizer_state": missing_quantizer_keys,
        "start_epoch": start_epoch,
    }
if args.resume and checkpoint_payload is not None:
    # Reconcile a crash after latest.pt publication but before best.pt/JSON publication.
    selected_resume = args.resume.resolve()
    if not latest_path.is_file() or selected_resume != latest_path.resolve():
        # An explicitly restored/periodic checkpoint becomes the single authoritative latest.
        atomic_torch_save(checkpoint_payload, latest_path)
    best_epoch = max(
        record["training"], key=lambda item: item["validation_global_sdr"]
    )["epoch"]
    existing_best_epoch = None
    if best_path.is_file():
        existing_best = load_checkpoint(best_path, "cpu")
        if existing_best.get("resolved_config") != config:
            raise ValueError("best checkpoint configuration differs from the resumed run")
        existing_best_epoch = existing_best.get("epoch")
    if existing_best_epoch != best_epoch:
        if checkpoint_payload.get("epoch") != best_epoch:
            raise ValueError("cannot reconcile best checkpoint with authoritative history")
        atomic_torch_save(checkpoint_payload, best_path)
record.setdefault("checkpoint_hashes", {})["latest"] = (
    sha256(latest_path) if latest_path.is_file() else None
)
record["checkpoint_hashes"]["best"] = sha256(best_path) if best_path.is_file() else None
record["checkpoint_sha256"] = record["checkpoint_hashes"]["latest"]
# Persist resume provenance and reconciliation even when training is already complete.
save_record(record_path, record)
weights = train_config
steps_per_epoch = math.ceil(config["data"]["epoch_chunks"] / weights["batch_size"])
global_step = (
    int(checkpoint_payload.get("global_step", start_epoch * steps_per_epoch))
    if args.resume and checkpoint_payload is not None
    else 0
)

for epoch in range(start_epoch, config["train"]["epochs"]):
    dataset.set_epoch(epoch)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_started = time.perf_counter()
    training_started = time.perf_counter()
    sums = {
        "loss": 0.0,
        "waveform_l1": 0.0,
        "complex_l1": 0.0,
        "sdr": 0.0,
        "gradient_norm": 0.0,
    }
    distillation_sum = 0.0
    distillation_batches = 0
    batches = 0
    skipped_optimizer_steps = 0
    optimizer_updates = 0
    for mixture, targets in loader:
        mixture, targets = mixture.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        teacher_targets = None
        if teacher is not None and global_step % distillation.every_n_steps == 0:
            teacher_targets = prepare_teacher_targets(
                teacher(mixture), mixture, distillation.enforce_mixture_consistency
            )
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_mode != "off"
            else nullcontext()
        )
        with autocast:
            estimate_spectra = model.spectrograms(mixture)
        estimates = model.stft.synthesis(estimate_spectra, mixture.shape[-1])
        target_spectra = model.stft.analysis(targets)
        waveform_loss = F.l1_loss(estimates, targets)
        spectrum_loss = complex_l1(estimate_spectra, target_spectra)
        loss = weights["waveform_l1"] * waveform_loss + weights["complex_l1"] * spectrum_loss
        if teacher_targets is not None:
            teacher_loss = waveform_distillation_l1(estimates, teacher_targets)
            loss = loss + distillation.weight * teacher_loss
            distillation_sum += float(teacher_loss.detach())
            distillation_batches += 1
        if weights.get("multires_stft", 0) > 0:
            loss = loss + weights["multires_stft"] * multiresolution_stft_loss(
                estimates, targets
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {batches}")
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        gradient_is_finite = bool(torch.isfinite(gradient_norm))
        if not gradient_is_finite and not scaler.is_enabled():
            raise FloatingPointError(
                f"non-finite gradient norm at epoch {epoch}, batch {batches}"
            )
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        step_was_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
        skipped_optimizer_steps += int(step_was_skipped)
        optimizer_updates += int(not step_was_skipped)
        global_step += 1
        sums["loss"] += float(loss.detach())
        sums["waveform_l1"] += float(waveform_loss.detach())
        sums["complex_l1"] += float(spectrum_loss.detach())
        sums["sdr"] += float(global_sdr(estimates.detach(), targets))
        sums["gradient_norm"] += (
            float(gradient_norm.detach()) if gradient_is_finite else 0.0
        )
        batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    metrics = {key: value / max(1, batches) for key, value in sums.items()}
    metrics["attempted_steps"] = batches
    metrics["optimizer_steps"] = optimizer_updates
    metrics["skipped_optimizer_steps"] = skipped_optimizer_steps
    metrics["training_seconds"] = training_seconds
    metrics["training_chunks_per_second"] = config["data"]["epoch_chunks"] / max(
        training_seconds, 1e-9
    )
    if teacher is not None:
        metrics["distillation_l1"] = distillation_sum / max(1, distillation_batches)
        metrics["distillation_batches"] = distillation_batches
    metrics["epoch"] = epoch
    model.eval()
    validation_started = time.perf_counter()
    diagnostics = DevelopmentDiagnostics(model_cfg.sources)
    with torch.inference_mode():
        for mixture, targets in validation_loader:
            mixture, targets = mixture.to(device), targets.to(device)
            estimates = model(mixture)
            diagnostics.update(estimates, targets, mixture)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    validation_seconds = time.perf_counter() - validation_started
    development = diagnostics.compute()
    metrics["validation_development_diagnostics"] = development
    metrics["validation_seconds"] = validation_seconds
    metrics["epoch_seconds"] = time.perf_counter() - epoch_started
    if device.type == "cuda":
        metrics["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        metrics["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
    # Preserve these flat names for existing records and best-checkpoint selection.
    metrics["validation_waveform_l1"] = sum(
        development["per_stem_waveform_l1"].values()
    ) / model_cfg.sources
    metrics["validation_global_sdr"] = development["global_sdr"]
    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
    if scheduler is not None:
        scheduler.step()
    metrics["next_learning_rate"] = optimizer.param_groups[0]["lr"]
    metrics["quantization"] = ternary_training_summary(model)
    if not all(
        math.isfinite(value)
        for key, value in metrics.items()
        if key not in {"validation_development_diagnostics", "quantization"}
        and isinstance(value, (int, float))
    ):
        raise FloatingPointError(f"non-finite epoch metrics at epoch {epoch}")
    record["training"].append(metrics)
    checkpoint = {
        "config": asdict(model_cfg),
        "resolved_config": config,
        "epoch": epoch,
        "global_step": global_step,
        "data_workers": args.workers,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "metrics": metrics,
        "training_history": record["training"],
    }
    atomic_torch_save(checkpoint, latest_path)
    previous = record["training"][:-1]
    if not previous or metrics["validation_global_sdr"] > max(
        item["validation_global_sdr"] for item in previous
    ):
        atomic_torch_save(checkpoint, best_path)
    checkpoint_every = int(config["train"].get("checkpoint_every", 0))
    if checkpoint_every > 0 and (
        (epoch + 1) % checkpoint_every == 0 or epoch + 1 == config["train"]["epochs"]
    ):
        atomic_torch_save(checkpoint, args.output_dir / f"epoch-{epoch + 1:04d}.pt")
    record.setdefault("checkpoint_hashes", {})["latest"] = sha256(latest_path)
    record["checkpoint_hashes"]["best"] = sha256(best_path) if best_path.is_file() else None
    record["checkpoint_sha256"] = record["checkpoint_hashes"]["latest"]
    save_record(record_path, record)
    print(json.dumps(metrics))
