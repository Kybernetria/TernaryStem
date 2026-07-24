import json

import pytest

from ternarystem.deployment import BillingLedger, RungState, SyncReceipt


def test_billing_ledger_is_cumulative_hash_chained_and_persistent(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = BillingLedger(path)
    first = ledger.append(category="setup", duration_seconds=1800, deployment_id="d2")
    second = ledger.append(category="training", duration_seconds=3600, deployment_id="d2")
    assert first["cumulative_seconds"] == 1800
    assert second["cumulative_seconds"] == 5400
    assert second["previous_sha256"] == first["event_sha256"]
    assert BillingLedger(path).totals().seconds == 5400
    assert BillingLedger(path).totals().cents == 120


def test_billing_ledger_corruption_and_absolute_cutoff_fail_closed(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = BillingLedger(path)
    ledger.append(category="idle", duration_seconds=1, deployment_id="d2")
    payload = json.loads(path.read_text())
    payload["events"][0]["duration_seconds"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash chain"):
        ledger.totals()

    cutoff = BillingLedger(tmp_path / "cutoff.json")
    cutoff.append(category="training", duration_seconds=179_999, deployment_id="d2")
    with pytest.raises(ValueError, match="absolute"):
        cutoff.assert_within_absolute(1)
    assert not cutoff.has_planned_reserve(1)


def test_rung_cannot_promote_before_verified_remote_commit(tmp_path):
    state = RungState(tmp_path / "state.json")
    payload = state.initialize(deployment_id="d2", candidate_id="A", rung_id="10k")
    payload = state.transition("running", expected_revision=payload["revision"])
    payload = state.transition(
        "checkpointed", expected_revision=payload["revision"], checkpoint_sha256="abc"
    )
    with pytest.raises(ValueError, match="invalid rung transition"):
        state.transition("gate_passed", expected_revision=payload["revision"])
    receipt = SyncReceipt("d2", "A", "10k", "abc", "manifest", "generation-1", True)
    payload = state.transition(
        "remotely_committed", expected_revision=payload["revision"], receipt=receipt
    )
    payload = state.transition("gate_passed", expected_revision=payload["revision"])
    assert payload["state"] == "gate_passed"


def test_rung_refuses_unverified_or_conflicting_receipt_and_restart_revision(tmp_path):
    state = RungState(tmp_path / "state.json")
    state.initialize(deployment_id="d2", candidate_id="B", rung_id="30k")
    state.transition("running", expected_revision=0)
    state.transition("checkpointed", expected_revision=1, checkpoint_sha256="good")
    bad = SyncReceipt("d2", "B", "30k", "bad", "manifest", "generation", True)
    with pytest.raises(ValueError, match="conflicts"):
        state.transition("remotely_committed", expected_revision=2, receipt=bad)
    unverified = SyncReceipt("d2", "B", "30k", "good", "manifest", "generation", False)
    with pytest.raises(ValueError, match="verified"):
        state.transition("remotely_committed", expected_revision=2, receipt=unverified)
    with pytest.raises(ValueError, match="revision conflict"):
        RungState(state.path).transition("stopped", expected_revision=1)
