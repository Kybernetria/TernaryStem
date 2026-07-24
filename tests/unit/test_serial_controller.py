import json

import pytest

from ternarystem.deployment.controller import SerialRungController, publish_local_rung
from ternarystem.deployment.gates import load_gate_policy
from ternarystem.deployment.ledger import BillingLedger


POLICY = load_gate_policy("configs/deployment_2/gate_policy.yaml")


def make_controller(tmp_path):
    return SerialRungController(
        deployment_id="d2",
        candidate_id="A",
        rung_id="10k",
        run_root=tmp_path / "runs",
        remote_root=tmp_path / "remote",
        ledger=BillingLedger.from_gate_policy(tmp_path / "ledger.json", POLICY),
        gate_policy=POLICY,
    )


def trainer_factory(calls):
    def trainer(checkpoint, manifest):
        calls.append("train")
        publish_local_rung(
            checkpoint,
            manifest,
            b"checkpoint",
            {
                "deployment_id": "d2",
                "candidate_id": "A",
                "rung_id": "10k",
                "completed_chunks": 10_000,
            },
        )

    return trainer


@pytest.mark.parametrize(
    "phase",
    [
        "after_trainer_before_checkpoint_state",
        "after_checkpoint_state_before_billing",
        "after_checkpoint_state",
        "before_staging",
        "after_staging",
        "after_generation_publish",
        "after_receipt_publish",
        "after_remote_commit_state",
    ],
)
def test_every_transaction_crash_point_resumes_without_retraining_or_duplicate_billing(
    tmp_path, phase
):
    calls = []
    controller = make_controller(tmp_path)

    def crash(current):
        if current == phase:
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected"):
        controller.execute(
            trainer=trainer_factory(calls),
            decision=lambda manifest: True,
            billed_seconds=60,
            fault_hook=crash,
        )
    result = make_controller(tmp_path).execute(
        trainer=trainer_factory(calls),
        decision=lambda manifest: True,
        billed_seconds=60,
    )
    assert result["state"] == "gate_passed"
    assert calls == ["train"]
    ledger = BillingLedger(tmp_path / "ledger.json")
    assert ledger.totals().seconds == 60
    assert len(ledger.read()) == 1
    assert len(list((tmp_path / "remote/generations").iterdir())) == 1
    assert len(list((tmp_path / "remote/receipts").iterdir())) == 1


def test_checkpointed_resume_does_not_project_training_cost_twice(tmp_path):
    calls = []
    controller = make_controller(tmp_path)

    def crash(phase):
        if phase == "after_checkpoint_state":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected"):
        controller.execute(
            trainer=trainer_factory(calls),
            decision=lambda manifest: True,
            billed_seconds=60,
            fault_hook=crash,
        )
    controller.ledger.append(category="idle", duration_seconds=26_800, deployment_id="d2")
    result = make_controller(tmp_path).execute(
        trainer=trainer_factory(calls),
        decision=lambda manifest: True,
        billed_seconds=60,
    )
    assert result["state"] == "gate_passed"
    assert calls == ["train"]


def test_controller_refuses_rung_when_policy_reserve_does_not_fit(tmp_path):
    controller = make_controller(tmp_path)
    controller.ledger.append(category="setup", duration_seconds=39_270, deployment_id="d2")
    assert controller.ledger.has_planned_reserve(60)
    with pytest.raises(ValueError, match="planned deployment budget"):
        controller.execute(
            trainer=trainer_factory([]),
            decision=lambda manifest: True,
            billed_seconds=60,
        )


def test_failed_gate_stops_and_never_reexecutes(tmp_path):
    calls = []
    controller = make_controller(tmp_path)
    result = controller.execute(
        trainer=trainer_factory(calls),
        decision=lambda manifest: False,
        billed_seconds=60,
    )
    assert result["state"] == "stopped"
    assert controller.execute(
        trainer=trainer_factory(calls),
        decision=lambda manifest: True,
        billed_seconds=60,
    )["state"] == "stopped"
    assert calls == ["train"]


def test_partial_or_corrupt_local_artifacts_fail_closed(tmp_path):
    controller = make_controller(tmp_path)
    controller.root.mkdir(parents=True)
    controller.state.initialize(deployment_id="d2", candidate_id="A", rung_id="10k")
    controller.state.transition("running", expected_revision=0)
    controller.checkpoint.write_bytes(b"partial")
    with pytest.raises(ValueError, match="partial or conflicting"):
        controller.execute(
            trainer=lambda checkpoint, manifest: None,
            decision=lambda manifest: True,
            billed_seconds=1,
        )


def test_manifest_lineage_and_hash_conflicts_fail_closed(tmp_path):
    calls = []
    controller = make_controller(tmp_path)

    def crash(phase):
        if phase == "after_trainer_before_checkpoint_state":
            raise RuntimeError

    with pytest.raises(RuntimeError):
        controller.execute(
            trainer=trainer_factory(calls),
            decision=lambda manifest: True,
            billed_seconds=1,
            fault_hook=crash,
        )
    manifest = json.loads(controller.manifest_path.read_text())
    manifest["candidate_id"] = "B"
    controller.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partial or conflicting"):
        controller.execute(
            trainer=trainer_factory(calls),
            decision=lambda manifest: True,
            billed_seconds=1,
        )
