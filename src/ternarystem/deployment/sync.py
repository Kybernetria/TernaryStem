"""Atomic immutable-generation synchronization for deployment #2 artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from ternarystem.training import atomic_json_save

FaultHook = Callable[[str], None]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactSynchronizer:
    """Publish a verified generation, then atomically publish its receipt last."""

    def __init__(self, remote_root: str | Path) -> None:
        self.remote_root = Path(remote_root)
        self.generations = self.remote_root / "generations"
        self.receipts = self.remote_root / "receipts"
        self.lock_path = self.remote_root / ".sync.lock"

    def publish(
        self,
        *,
        deployment_id: str,
        candidate_id: str,
        rung_id: str,
        checkpoint: str | Path,
        manifest: dict,
        fault_hook: FaultHook | None = None,
    ) -> dict:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise ValueError("checkpoint to synchronize is missing")
        checkpoint_hash = sha256(checkpoint)
        manifest = dict(manifest)
        if manifest.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("manifest checkpoint hash does not match artifact")
        lineage = f"{deployment_id}-{candidate_id}-{rung_id}"
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        generation_id = f"{lineage}-{checkpoint_hash[:16]}-{manifest_hash[:16]}"
        generation = self.generations / generation_id
        receipt_path = self.receipts / f"{lineage}.json"

        self.remote_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("artifact synchronization is already active") from error
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("generation_id") == generation_id
                    and receipt.get("checkpoint_sha256") == checkpoint_hash
                    and receipt.get("manifest_sha256") == manifest_hash
                    and generation.is_dir()
                ):
                    self._verify_generation(generation, checkpoint_hash, manifest_hash)
                    return receipt
                raise ValueError("a conflicting synchronization receipt already exists")
            if fault_hook:
                fault_hook("before_staging")
            self.generations.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=self.generations))
            try:
                shutil.copyfile(checkpoint, staging / "checkpoint.pt")
                (staging / "manifest.json").write_bytes(manifest_bytes + b"\n")
                if fault_hook:
                    fault_hook("after_staging")
                self._verify_generation(staging, checkpoint_hash, manifest_hash)
                if generation.exists():
                    self._verify_generation(generation, checkpoint_hash, manifest_hash)
                    shutil.rmtree(staging)
                else:
                    os.replace(staging, generation)
                if fault_hook:
                    fault_hook("after_generation_publish")
                receipt = {
                    "schema_version": 1,
                    "deployment_id": deployment_id,
                    "candidate_id": candidate_id,
                    "rung_id": rung_id,
                    "generation_id": generation_id,
                    "checkpoint_sha256": checkpoint_hash,
                    "manifest_sha256": manifest_hash,
                    "independently_verified": True,
                }
                atomic_json_save(receipt, receipt_path)
                if fault_hook:
                    fault_hook("after_receipt_publish")
                return receipt
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _verify_generation(generation: Path, checkpoint_hash: str, manifest_hash: str) -> None:
        checkpoint = generation / "checkpoint.pt"
        manifest = generation / "manifest.json"
        if not checkpoint.is_file() or not manifest.is_file():
            raise ValueError("remote generation is incomplete")
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError("remote checkpoint checksum verification failed")
        canonical = json.dumps(
            json.loads(manifest.read_text(encoding="utf-8")),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != manifest_hash:
            raise ValueError("remote manifest checksum verification failed")
