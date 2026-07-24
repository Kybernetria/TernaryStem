import math

import pytest

from ternarystem.deployment.gates import (
    candidate_100k_decision,
    load_gate_policy,
    paired_track_bootstrap,
    recent_ols_slope,
)

POLICY_PATH = "configs/deployment_2/gate_policy.yaml"


def record(values):
    return {"training": [{"validation_global_sdr": value} for value in values]}


def test_recent_slope_and_100k_gate_boundaries():
    policy = load_gate_policy(POLICY_PATH)
    assert recent_ols_slope([1.0, 2.0, 3.0], 3) == 1.0
    passed = candidate_100k_decision(record([5.1, 5.3, 5.4]), 5.0, policy)
    assert passed["decision"] == "promote"
    assert passed["quality_pass"] and passed["slope_pass"]
    flat = candidate_100k_decision(record([5.4, 5.4, 5.4]), 5.0, policy)
    assert flat["decision"] == "stop"
    assert flat["quality_pass"] and not flat["slope_pass"]


@pytest.mark.parametrize("missing", [None, math.nan, math.inf])
def test_100k_gate_refuses_missing_or_nonfinite(missing):
    policy = load_gate_policy(POLICY_PATH)
    with pytest.raises(ValueError, match="missing or non-finite"):
        candidate_100k_decision(record([5.0, 5.2, missing]), 5.0, policy)


def test_paired_bootstrap_quality_winner_and_tiebreak_labels():
    policy = load_gate_policy(POLICY_PATH)
    tracks = {f"track-{index}": 6.0 for index in range(14)}
    weaker = {name: value - 0.2 for name, value in tracks.items()}
    quality = paired_track_bootstrap(tracks, weaker, policy)
    assert quality["winner"] == "A"
    assert quality["label"] == "quality_winner"
    tied = paired_track_bootstrap(tracks, dict(tracks), policy)
    assert tied["winner"] == "A"
    assert tied["label"] == "deployment_tiebreak_not_quality_winner"


def test_paired_bootstrap_refuses_unpaired_tracks():
    policy = load_gate_policy(POLICY_PATH)
    with pytest.raises(ValueError, match="unpaired"):
        paired_track_bootstrap({"a": 1.0}, {"b": 1.0}, policy)
