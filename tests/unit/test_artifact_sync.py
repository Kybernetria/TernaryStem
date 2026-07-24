import fcntl
import json
import os

import pytest

from ternarystem.deployment.sync import ArtifactSynchronizer, sha256


def inputs(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    manifest = {
        "deployment_id": "d2",
        "candidate_id": "A",
        "rung_id": "10k",
        "checkpoint_sha256": sha256(checkpoint),
        "completed_chunks": 10_000,
    }
    return checkpoint, manifest


def test_receipt_is_published_last_and_restart_is_idempotent(tmp_path):
    checkpoint, manifest = inputs(tmp_path)
    remote = tmp_path / "remote"
    synchronizer = ArtifactSynchronizer(remote)
    receipt = synchronizer.publish(
        deployment_id="d2",
        candidate_id="A",
        rung_id="10k",
        checkpoint=checkpoint,
        manifest=manifest,
    )
    assert receipt["independently_verified"] is True
    assert (remote / "generations" / receipt["generation_id"]).is_dir()
    assert json.loads((remote / "receipts/d2-A-10k.json").read_text()) == receipt
    assert synchronizer.publish(
        deployment_id="d2",
        candidate_id="A",
        rung_id="10k",
        checkpoint=checkpoint,
        manifest=manifest,
    ) == receipt


@pytest.mark.parametrize("phase", ["before_staging", "after_staging"])
def test_crash_before_generation_publish_never_creates_receipt(tmp_path, phase):
    checkpoint, manifest = inputs(tmp_path)
    remote = tmp_path / "remote"

    def crash(current):
        if current == phase:
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected"):
        ArtifactSynchronizer(remote).publish(
            deployment_id="d2",
            candidate_id="A",
            rung_id="10k",
            checkpoint=checkpoint,
            manifest=manifest,
            fault_hook=crash,
        )
    assert not (remote / "receipts/d2-A-10k.json").exists()
    assert not [path for path in (remote / "generations").glob("*") if path.is_dir()]


def test_crash_after_generation_before_receipt_recovers_without_duplicate(tmp_path):
    checkpoint, manifest = inputs(tmp_path)
    remote = tmp_path / "remote"

    def crash(phase):
        if phase == "after_generation_publish":
            raise RuntimeError("injected crash")

    synchronizer = ArtifactSynchronizer(remote)
    with pytest.raises(RuntimeError, match="injected"):
        synchronizer.publish(
            deployment_id="d2",
            candidate_id="A",
            rung_id="10k",
            checkpoint=checkpoint,
            manifest=manifest,
            fault_hook=crash,
        )
    assert len(list((remote / "generations").iterdir())) == 1
    assert not (remote / "receipts/d2-A-10k.json").exists()
    receipt = synchronizer.publish(
        deployment_id="d2",
        candidate_id="A",
        rung_id="10k",
        checkpoint=checkpoint,
        manifest=manifest,
    )
    assert len(list((remote / "generations").iterdir())) == 1
    assert receipt["independently_verified"] is True


def test_concurrent_synchronization_is_refused(tmp_path):
    checkpoint, manifest = inputs(tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    descriptor = os.open(remote / ".sync.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            ArtifactSynchronizer(remote).publish(
                deployment_id="d2",
                candidate_id="A",
                rung_id="10k",
                checkpoint=checkpoint,
                manifest=manifest,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_conflicting_receipt_refuses_overwrite(tmp_path):
    checkpoint, manifest = inputs(tmp_path)
    remote = tmp_path / "remote"
    synchronizer = ArtifactSynchronizer(remote)
    synchronizer.publish(
        deployment_id="d2",
        candidate_id="A",
        rung_id="10k",
        checkpoint=checkpoint,
        manifest=manifest,
    )
    checkpoint.write_bytes(b"different")
    manifest["checkpoint_sha256"] = sha256(checkpoint)
    with pytest.raises(ValueError, match="conflicting"):
        synchronizer.publish(
            deployment_id="d2",
            candidate_id="A",
            rung_id="10k",
            checkpoint=checkpoint,
            manifest=manifest,
        )
