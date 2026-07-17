#!/usr/bin/env python3
"""Measure immediate development-set impact of quantizing one layer family at a time.

This command only opens the MUSDB training partition and uses the frozen validation
track list. It does not train, recover, or access the official test partition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ternarystem.data import MUSDBChunkDataset, validate_track_names
from ternarystem.evaluation import base_record, save_record, sha256
from ternarystem.losses import complex_l1, global_sdr
from ternarystem.models import LAYER_FAMILIES, Separator, SeparatorConfig
from ternarystem.quant import symmetric_weight_values, ternary_values

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path, help="FP32 checkpoint")
parser.add_argument("--data-root", type=Path, required=True, help="MUSDB18-HQ train directory")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--precision", choices=("ternary", "w4a8", "w8a8"), default="ternary")
parser.add_argument("--families", nargs="+", choices=sorted(LAYER_FAMILIES), default=sorted(LAYER_FAMILIES))
parser.add_argument("--chunks", type=int, default=28)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
if "config" not in payload:
    raise SystemExit("checkpoint must contain its resolved SeparatorConfig")
raw_config = dict(payload["config"])
raw_config["channels"] = tuple(raw_config["channels"])
raw_config.setdefault("layer_precisions", {})
fp_config = SeparatorConfig(**raw_config)
if fp_config.layer_precisions:
    raise SystemExit("sensitivity requires an FP checkpoint with empty layer_precisions")
state_dict = payload["state_dict"]
seed = int(payload.get("resolved_config", {}).get("seed", 20250218))
torch.manual_seed(seed)
track_names = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
_, validation_names = validate_track_names(track_names)
chunk_samples = round(44100 * float(payload.get("resolved_config", {}).get("data", {}).get("chunk_seconds", 6.0)))
dataset = MUSDBChunkDataset(
    args.data_root,
    validation_names,
    chunk_samples,
    args.chunks,
    seed + 1,
    remix=False,
    augment=False,
)
loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
device = torch.device(args.device)


def evaluate(model: Separator) -> dict[str, float]:
    model.eval()
    sums = {"diagnostic_loss": 0.0, "waveform_l1": 0.0, "complex_l1": 0.0, "global_sdr": 0.0}
    batches = 0
    with torch.inference_mode():
        for mixture, targets in loader:
            mixture, targets = mixture.to(device), targets.to(device)
            spectra = model.spectrograms(mixture)
            estimates = model.stft.synthesis(spectra, mixture.shape[-1])
            target_spectra = model.stft.analysis(targets)
            waveform_loss = F.l1_loss(estimates, targets)
            spectrum_loss = complex_l1(spectra, target_spectra)
            sums["diagnostic_loss"] += float(waveform_loss + spectrum_loss)
            sums["waveform_l1"] += float(waveform_loss)
            sums["complex_l1"] += float(spectrum_loss)
            sums["global_sdr"] += float(global_sdr(estimates, targets))
            batches += 1
    return {key: value / max(1, batches) for key, value in sums.items()}


def family_for(name: str) -> str | None:
    if name in {"network.input_projection.weight", "network.output_projection.weight"}:
        return "projections"
    if ".tdf.layers." in name:
        return "tdf_linear"
    if name.startswith("network.decoder."):
        return "decoder_conv"
    if name.startswith(f"network.encoder.{len(fp_config.channels) - 1}."):
        return "bottleneck_conv"
    if name.startswith("network.encoder."):
        return "encoder_conv"
    return None


def quant_stats(family: str, precision: str) -> tuple[list[dict], int]:
    statistics = []
    covered = 0
    for name, tensor in sorted(state_dict.items()):
        if not name.endswith("weight") or tensor.ndim < 2 or family_for(name) != family:
            continue
        covered += tensor.numel()
        if precision == "ternary":
            _, scale, values = ternary_values(
                tensor, method=fp_config.ternary_method, zero_ratio=fp_config.zero_ratio
            )
        else:
            bits = 4 if precision == "w4a8" else 8
            _, scale, values = symmetric_weight_values(
                tensor, bits=bits, group_size=fp_config.w4_group_size if bits == 4 else None
            )
        statistics.append(
            {
                "name": name,
                "parameters": tensor.numel(),
                "scale_mean": float(scale.mean()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "zero_fraction": float((values == 0).float().mean()),
                "integer_min": int(values.min()),
                "integer_max": int(values.max()),
            }
        )
    return statistics, covered


fp_model = Separator(fp_config).to(device)
fp_model.load_state_dict(state_dict)
baseline = evaluate(fp_model)
total_parameters = sum(parameter.numel() for parameter in fp_model.parameters())
eligible_parameters = sum(
    tensor.numel()
    for name, tensor in state_dict.items()
    if name.endswith("weight") and tensor.ndim >= 2 and family_for(name) is not None
)
resolved = {
    "checkpoint_config": asdict(fp_config),
    "precision": args.precision,
    "families": args.families,
    "chunks": args.chunks,
    "batch_size": args.batch_size,
    "validation_tracks": validation_names,
}
record = base_record(resolved, seed, args.device)
record["checkpoint_sha256"] = sha256(args.checkpoint)
record["sensitivity"] = {
    "evidence": "MUSDB18-HQ development-split immediate diagnostic; no QAT recovery",
    "baseline": baseline,
    "total_parameters": total_parameters,
    "eligible_weight_parameters": eligible_parameters,
    "families": [],
}
for family in args.families:
    config = replace(fp_config, layer_precisions={family: args.precision})
    model = Separator(config).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    metrics = evaluate(model)
    statistics, covered = quant_stats(family, args.precision)
    saturation = {
        name: float(module.activation_quant.saturation_rate)
        for name, module in model.named_modules()
        if hasattr(module, "activation_quant")
    }
    record["sensitivity"]["families"].append(
        {
            "family": family,
            "precision": args.precision,
            "covered_parameters": covered,
            "coverage_of_eligible": covered / max(1, eligible_parameters),
            "coverage_of_all_parameters": covered / max(1, total_parameters),
            "metrics": metrics,
            "delta": {key: metrics[key] - baseline[key] for key in metrics},
            "weight_statistics": statistics,
            "activation_saturation": saturation,
            "resolved_model_config": asdict(config),
        }
    )
save_record(args.output, record)
print(json.dumps(record["sensitivity"], indent=2))
