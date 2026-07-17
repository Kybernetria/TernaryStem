"""Summaries for development experiment JSON records."""

from __future__ import annotations


def _diagnostics(epoch: dict) -> dict | None:
    value = epoch.get("validation_development_diagnostics")
    if isinstance(value, dict):
        return value
    # Legacy records only have model diagnostics and cannot supply a baseline.
    if "validation_global_sdr" in epoch:
        return {
            "label": "development diagnostics (not BSSEval)",
            "global_sdr": epoch["validation_global_sdr"],
            "per_stem_global_sdr": None,
            "per_stem_waveform_l1": None,
            "equal_share_baseline": None,
        }
    return None


def summarize_record(record: dict, label: str) -> dict:
    history = record.get("training") or []
    scored = [item for item in history if "validation_global_sdr" in item]
    best = max(scored, key=lambda item: item["validation_global_sdr"]) if scored else None
    final = scored[-1] if scored else None
    config = record.get("config") or {}
    model = config.get("model") or {}
    precision = (config.get("quant") or {}).get("layer_precisions") or {}

    def point(item: dict | None) -> dict | None:
        if item is None:
            return None
        diagnostics = _diagnostics(item)
        baseline = diagnostics.get("equal_share_baseline") if diagnostics else None
        model_sdr = diagnostics.get("global_sdr") if diagnostics else None
        baseline_sdr = baseline.get("global_sdr") if baseline else None
        return {
            "epoch": item.get("epoch"),
            "development_global_sdr": model_sdr,
            "equal_share_development_global_sdr": baseline_sdr,
            "model_minus_equal_share_db": (
                model_sdr - baseline_sdr
                if model_sdr is not None and baseline_sdr is not None
                else None
            ),
            "per_stem_development_global_sdr": (
                diagnostics.get("per_stem_global_sdr") if diagnostics else None
            ),
        }

    return {
        "label": label,
        "metric_label": "development global_sdr (not BSSEval)",
        "output_parameterization": model.get("output_parameterization", "direct_estimate"),
        "training_precision": "fp32" if not precision else "quantization_aware_or_mixed",
        "layer_precisions": precision,
        "best": point(best),
        "final": point(final),
    }
