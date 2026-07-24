"""Machine-readable, fail-closed deployment #2 promotion gates."""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any

import yaml


def load_gate_policy(path) -> dict:
    with open(path, encoding="utf-8") as stream:
        policy = yaml.safe_load(stream)
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("gate policy schema is invalid")
    return policy


def require_finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"gate metric {name} is missing or non-finite")
    return float(value)


def recent_ols_slope(values: list[float], window: int) -> float:
    if len(values) < window or window < 2:
        raise ValueError("insufficient validation history for recent slope")
    selected = [require_finite(value, "validation_history") for value in values[-window:]]
    center = (window - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(window))
    return sum((index - center) * (value - mean(selected)) for index, value in enumerate(selected)) / denominator


def candidate_100k_decision(record: dict, best_30k: float, policy: dict) -> dict:
    history = record.get("training")
    if not isinstance(history, list) or not history:
        raise ValueError("candidate record has no training history")
    values = [require_finite(item.get("validation_global_sdr"), "validation_global_sdr") for item in history]
    best = max(values)
    best_30k = require_finite(best_30k, "best_30k")
    gate = policy["rungs"]["comparison_100k"]
    quality = best >= require_finite(gate["minimum_global_sdr"], "minimum_global_sdr") or (
        best - best_30k
        >= require_finite(gate["alternative_improvement_over_30k_db"], "improvement")
    )
    slope_policy = policy["recent_slope"]
    slope = recent_ols_slope(values, int(slope_policy["window"]))
    slope_pass = best >= policy["rungs"]["final"]["target_global_sdr"] or slope > require_finite(
        slope_policy["minimum_db_per_validation"], "minimum_slope"
    )
    return {
        "decision": "promote" if quality and slope_pass else "stop",
        "best_global_sdr": best,
        "improvement_over_30k_db": best - best_30k,
        "recent_ols_slope": slope,
        "quality_pass": quality,
        "slope_pass": slope_pass,
    }


def paired_track_bootstrap(
    candidate_a: dict[str, float], candidate_b: dict[str, float], policy: dict
) -> dict:
    tracks = sorted(set(candidate_a) & set(candidate_b))
    if not tracks or set(tracks) != set(candidate_a) or set(tracks) != set(candidate_b):
        raise ValueError("paired track metrics are missing or unpaired")
    differences = [
        require_finite(candidate_a[track], f"A:{track}")
        - require_finite(candidate_b[track], f"B:{track}")
        for track in tracks
    ]
    interval = policy["paired_interval"]
    samples = int(interval["bootstrap_samples"])
    confidence = require_finite(interval["confidence_level"], "confidence_level")
    if samples < 100 or not 0 < confidence < 1:
        raise ValueError("paired interval policy is invalid")
    rng = random.Random(int(interval["seed"]))
    bootstrap = sorted(
        mean(rng.choice(differences) for _ in differences) for _ in range(samples)
    )
    tail = (1 - confidence) / 2
    lower = bootstrap[min(samples - 1, max(0, math.floor(tail * samples)))]
    upper = bootstrap[min(samples - 1, max(0, math.ceil((1 - tail) * samples) - 1))]
    observed = mean(differences)
    threshold = require_finite(
        policy["quality_winner"]["minimum_paired_advantage_db"], "paired_advantage"
    )
    if observed >= threshold and lower > 0:
        winner = "A"
        label = "quality_winner"
    elif observed <= -threshold and upper < 0:
        winner = "B"
        label = "quality_winner"
    else:
        winner = "A"
        label = "deployment_tiebreak_not_quality_winner"
    return {
        "winner": winner,
        "label": label,
        "mean_paired_difference_a_minus_b_db": observed,
        "confidence_level": confidence,
        "interval_db": [lower, upper],
        "tracks": tracks,
    }
