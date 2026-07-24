"""Crash-safe serial rung state and synchronization receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ternarystem.training import atomic_json_save, canonical_json_sha256

TRANSITIONS = {
    "pending": {"running", "stopped"},
    "running": {"checkpointed", "stopped"},
    "checkpointed": {"remotely_committed", "stopped"},
    "remotely_committed": {"gate_passed", "stopped"},
    "gate_passed": set(),
    "stopped": set(),
}


@dataclass(frozen=True)
class SyncReceipt:
    deployment_id: str
    candidate_id: str
    rung_id: str
    checkpoint_sha256: str
    manifest_sha256: str
    remote_generation: str
    independently_verified: bool

    @property
    def receipt_sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


class RungState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self, *, deployment_id: str, candidate_id: str, rung_id: str) -> dict:
        if self.path.exists():
            raise ValueError("rung state already exists")
        payload = {
            "schema_version": 1,
            "deployment_id": deployment_id,
            "candidate_id": candidate_id,
            "rung_id": rung_id,
            "state": "pending",
            "checkpoint_sha256": None,
            "sync_receipt": None,
            "revision": 0,
        }
        atomic_json_save(payload, self.path)
        return self.read()

    def read(self) -> dict:
        if not self.path.is_file():
            raise ValueError("rung state is missing")
        try:
            import json

            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("rung state is unreadable or corrupt") from error
        required = {
            "schema_version",
            "deployment_id",
            "candidate_id",
            "rung_id",
            "state",
            "checkpoint_sha256",
            "sync_receipt",
            "revision",
        }
        if set(payload) != required or payload["schema_version"] != 1:
            raise ValueError("rung state schema is invalid")
        if payload["state"] not in TRANSITIONS or not isinstance(payload["revision"], int):
            raise ValueError("rung state value is invalid")
        receipt = payload["sync_receipt"]
        if receipt is not None:
            claimed = receipt.get("receipt_sha256")
            raw = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            if claimed != canonical_json_sha256(raw):
                raise ValueError("synchronization receipt hash is invalid")
        return payload

    def transition(
        self,
        target: str,
        *,
        expected_revision: int,
        checkpoint_sha256: str | None = None,
        receipt: SyncReceipt | None = None,
    ) -> dict:
        payload = self.read()
        if payload["revision"] != expected_revision:
            raise ValueError("rung state revision conflict")
        if target not in TRANSITIONS[payload["state"]]:
            raise ValueError(f"invalid rung transition {payload['state']} -> {target}")
        if target == "checkpointed":
            if not checkpoint_sha256:
                raise ValueError("checkpointed state requires a checkpoint hash")
            payload["checkpoint_sha256"] = checkpoint_sha256
        if target == "remotely_committed":
            if receipt is None or not receipt.independently_verified:
                raise ValueError("remote commit requires an independently verified receipt")
            if receipt.checkpoint_sha256 != payload["checkpoint_sha256"]:
                raise ValueError("synchronization receipt checkpoint hash conflicts")
            for key in ("deployment_id", "candidate_id", "rung_id"):
                if getattr(receipt, key) != payload[key]:
                    raise ValueError("synchronization receipt lineage conflicts")
            receipt_payload = asdict(receipt)
            receipt_payload["receipt_sha256"] = receipt.receipt_sha256
            payload["sync_receipt"] = receipt_payload
        if target == "gate_passed" and payload["sync_receipt"] is None:
            raise ValueError("an unsynchronized checkpoint cannot pass a gate")
        payload["state"] = target
        payload["revision"] += 1
        atomic_json_save(payload, self.path)
        return self.read()
