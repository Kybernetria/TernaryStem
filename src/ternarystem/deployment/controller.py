"""Restartable serial rung transaction used by the deployment #2 runner."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

from ternarystem.deployment.ledger import BillingLedger
from ternarystem.deployment.state import RungState, SyncReceipt
from ternarystem.deployment.sync import ArtifactSynchronizer, sha256
from ternarystem.training import atomic_json_save, canonical_json_sha256

Trainer = Callable[[Path, Path], None]
Decision = Callable[[dict], bool]
FaultHook = Callable[[str], None]


class SerialRungController:
    def __init__(
        self,
        *,
        deployment_id: str,
        candidate_id: str,
        rung_id: str,
        run_root: str | Path,
        remote_root: str | Path,
        ledger: BillingLedger,
        gate_policy: dict,
    ) -> None:
        self.deployment_id = deployment_id
        self.candidate_id = candidate_id
        self.rung_id = rung_id
        self.root = Path(run_root) / candidate_id / rung_id
        self.checkpoint = self.root / "checkpoint.pt"
        self.manifest_path = self.root / "manifest.json"
        self.state = RungState(self.root / "state.json")
        self.sync = ArtifactSynchronizer(remote_root)
        ledger.assert_matches_gate_policy(gate_policy)
        reserve = gate_policy.get("reserve") if isinstance(gate_policy, dict) else None
        if (
            not isinstance(reserve, dict)
            or not isinstance(reserve.get("overhead_fraction"), (int, float))
            or isinstance(reserve.get("overhead_fraction"), bool)
            or not 0 <= reserve["overhead_fraction"] <= 1
            or not isinstance(reserve.get("synchronization_seconds"), int)
            or isinstance(reserve.get("synchronization_seconds"), bool)
            or reserve["synchronization_seconds"] < 0
        ):
            raise ValueError("gate policy reserve is invalid")
        self.reserve_overhead = float(reserve["overhead_fraction"])
        self.synchronization_seconds = reserve["synchronization_seconds"]
        self.ledger = ledger

    def execute(
        self,
        *,
        trainer: Trainer,
        decision: Decision,
        billed_seconds: int,
        fault_hook: FaultHook | None = None,
    ) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.state.path.exists():
            payload = self.state.initialize(
                deployment_id=self.deployment_id,
                candidate_id=self.candidate_id,
                rung_id=self.rung_id,
            )
        else:
            payload = self.state.read()
        if payload["state"] in {"gate_passed", "stopped"}:
            return payload
        local_training_complete = payload["state"] == "running" and self._valid_local_checkpoint()
        needs_training = payload["state"] == "pending" or (
            payload["state"] == "running" and not local_training_complete
        )
        if needs_training:
            self.ledger.assert_within_absolute(billed_seconds)
            required_reserve = (
                math.ceil(billed_seconds * (1 + self.reserve_overhead))
                + self.synchronization_seconds
            )
            if not self.ledger.has_planned_reserve(required_reserve):
                raise ValueError("planned deployment budget has insufficient reserve")
        elif payload["state"] == "running":
            # Training completed before a crash but has not been billed yet.
            self.ledger.assert_within_absolute(billed_seconds)
        if payload["state"] == "pending":
            payload = self.state.transition("running", expected_revision=payload["revision"])
        billing_transaction = (
            f"{self.deployment_id}:{self.candidate_id}:{self.rung_id}:training"
        )
        if payload["state"] == "running":
            if (self.checkpoint.exists() or self.manifest_path.exists()) and not self._valid_local_checkpoint():
                raise ValueError("partial or conflicting local rung artifacts")
            if not self._valid_local_checkpoint():
                trainer(self.checkpoint, self.manifest_path)
                if fault_hook:
                    fault_hook("after_trainer_before_checkpoint_state")
            manifest = self._read_manifest()
            payload = self.state.transition(
                "checkpointed",
                expected_revision=payload["revision"],
                checkpoint_sha256=manifest["checkpoint_sha256"],
            )
            if fault_hook:
                fault_hook("after_checkpoint_state_before_billing")
        if payload["state"] == "checkpointed":
            if not self.ledger.has_transaction(billing_transaction):
                self.ledger.assert_within_absolute(billed_seconds)
            self.ledger.append(
                category="training",
                duration_seconds=billed_seconds,
                deployment_id=self.deployment_id,
                transaction_id=billing_transaction,
            )
            if fault_hook:
                fault_hook("after_checkpoint_state")
            manifest = self._read_manifest()
            receipt = self.sync.publish(
                deployment_id=self.deployment_id,
                candidate_id=self.candidate_id,
                rung_id=self.rung_id,
                checkpoint=self.checkpoint,
                manifest=manifest,
                fault_hook=fault_hook,
            )
            sync_receipt = SyncReceipt(
                self.deployment_id,
                self.candidate_id,
                self.rung_id,
                receipt["checkpoint_sha256"],
                receipt["manifest_sha256"],
                receipt["generation_id"],
                receipt["independently_verified"],
            )
            payload = self.state.transition(
                "remotely_committed",
                expected_revision=payload["revision"],
                receipt=sync_receipt,
            )
            if fault_hook:
                fault_hook("after_remote_commit_state")
        if payload["state"] == "remotely_committed":
            manifest = self._read_manifest()
            passed = bool(decision(manifest))
            payload = self.state.transition(
                "gate_passed" if passed else "stopped",
                expected_revision=payload["revision"],
            )
        return payload

    def _read_manifest(self) -> dict:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("local rung manifest is missing or corrupt") from error
        required = {
            "deployment_id": self.deployment_id,
            "candidate_id": self.candidate_id,
            "rung_id": self.rung_id,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise ValueError("local rung manifest lineage conflicts")
        if manifest.get("checkpoint_sha256") != sha256(self.checkpoint):
            raise ValueError("local rung checkpoint checksum conflicts")
        claimed = manifest.get("manifest_payload_sha256")
        raw = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
        if claimed != canonical_json_sha256(raw):
            raise ValueError("local rung manifest payload hash conflicts")
        return manifest

    def _valid_local_checkpoint(self) -> bool:
        if not self.checkpoint.is_file() or not self.manifest_path.is_file():
            return False
        try:
            self._read_manifest()
        except ValueError:
            return False
        return True


def publish_local_rung(checkpoint: Path, manifest_path: Path, payload: bytes, metadata: dict) -> None:
    """Fake-trainer helper used by the controller crash matrix."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(checkpoint)
    manifest = dict(metadata)
    manifest["checkpoint_sha256"] = sha256(checkpoint)
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    atomic_json_save(manifest, manifest_path)
