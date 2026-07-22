#!/usr/bin/env python3
"""Gate an FP run and materialize matched FP-control/selective-QAT configs."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import yaml

from ternarystem.evaluation import sha256
from ternarystem.models import LAYER_FAMILIES

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--sensitivity", type=Path, required=True)
parser.add_argument("--control-output", type=Path, required=True)
parser.add_argument("--qat-output", type=Path, required=True)
parser.add_argument("--families", nargs="+", required=True)
parser.add_argument("--min-fp-sdr", type=float, default=7.5)
parser.add_argument("--max-sensitivity-sdr-drop", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--checkpoint-every", type=int, default=5)
parser.add_argument("--allow-below-fp-gate", action="store_true")
args = parser.parse_args()

unknown = set(args.families) - LAYER_FAMILIES
if unknown:
    raise SystemExit(f"unknown QAT families: {sorted(unknown)}")
if "projections" in args.families:
    raise SystemExit("projections must remain FP32 in this selective-QAT pipeline")
record = json.loads(args.experiment.read_text(encoding="utf-8"))
if not record.get("training"):
    raise SystemExit("FP experiment contains no completed epochs")
best = max(record["training"], key=lambda item: item["validation_global_sdr"])
best_sdr = float(best["validation_global_sdr"])
diagnostics = best.get("validation_development_diagnostics", {})
baseline_sdr = float(diagnostics.get("equal_share_baseline", {}).get("global_sdr", -math.inf))
if not math.isfinite(best_sdr):
    raise SystemExit("FP best SDR is not finite")
failures = []
if best_sdr < args.min_fp_sdr:
    failures.append(f"best FP SDR {best_sdr:.4f} < required {args.min_fp_sdr:.4f}")
if best_sdr <= baseline_sdr:
    failures.append(f"best FP SDR {best_sdr:.4f} does not beat equal-share {baseline_sdr:.4f}")
if failures and not args.allow_below_fp_gate:
    raise SystemExit("FP GATE FAILED: " + "; ".join(failures))

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
if checkpoint.get("resolved_config") != record.get("config"):
    raise SystemExit("best checkpoint config does not match the FP experiment record")
if checkpoint.get("epoch") != best.get("epoch"):
    raise SystemExit("best checkpoint epoch does not match the best recorded metric")
checkpoint_metric = checkpoint.get("metrics", {}).get("validation_global_sdr")
if checkpoint_metric != best.get("validation_global_sdr"):
    raise SystemExit("best checkpoint metric does not match the experiment record")
checkpoint_hash = sha256(args.checkpoint)
recorded_hash = record.get("checkpoint_hashes", {}).get("best")
if recorded_hash != checkpoint_hash:
    raise SystemExit("best checkpoint hash does not match the experiment record")
sensitivity = json.loads(args.sensitivity.read_text(encoding="utf-8"))
if sensitivity.get("checkpoint_sha256") != checkpoint_hash:
    raise SystemExit("sensitivity was not computed from this exact best checkpoint")
by_family = {
    item["family"]: item for item in sensitivity.get("sensitivity", {}).get("families", [])
}
for family in args.families:
    if family not in by_family:
        raise SystemExit(f"sensitivity result is missing selected family: {family}")
    delta = float(by_family[family]["delta"]["global_sdr"])
    if not math.isfinite(delta) or delta < -args.max_sensitivity_sdr_drop:
        raise SystemExit(
            f"selected family {family} sensitivity {delta:.4f} dB exceeds "
            f"allowed {-args.max_sensitivity_sdr_drop:.4f} dB"
        )

base = copy.deepcopy(record["config"])
base.setdefault("quant", {})
base.setdefault("train", {})
base["train"].update(
    {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "checkpoint_every": args.checkpoint_every,
        "scheduler": {"name": "cosine", "t_max": args.epochs, "eta_min": 1e-6},
    }
)
control = copy.deepcopy(base)
control["quant"]["layer_precisions"] = {}
qat = copy.deepcopy(base)
qat["quant"]["layer_precisions"] = {family: "ternary" for family in args.families}
for path, payload in ((args.control_output, control), (args.qat_output, qat)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

print(
    json.dumps(
        {
            "fp_gate_overridden": bool(failures),
            "fp_gate_failures": failures,
            "best_fp_global_sdr": best_sdr,
            "equal_share_global_sdr": baseline_sdr,
            "checkpoint_sha256": checkpoint_hash,
            "selected_families": args.families,
            "control_config": str(args.control_output),
            "qat_config": str(args.qat_output),
        },
        indent=2,
    )
)
