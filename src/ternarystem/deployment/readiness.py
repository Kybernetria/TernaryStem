"""Exact-hash offline readiness checks for GPU deployment #2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml

from ternarystem.config import load_config
from ternarystem.models import architecture_identity, build_separator
from ternarystem.training import canonical_json_sha256


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def readiness_matrix(
    repository: str | Path,
    allowlist_path: str | Path,
    cuda_evidence_path: str | Path | None = None,
) -> dict:
    root = Path(repository).resolve()
    allowlist_file = root / allowlist_path
    allowlist = yaml.safe_load(allowlist_file.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("allowlist_schema", allowlist.get("schema_version") == 1, "schema_version must be 1")
    actual_commit = _git("-C", str(root), "rev-parse", "HEAD")
    lock_value = allowlist.get("source_commit_file")
    lock_path = root / lock_value if isinstance(lock_value, str) else None
    source_lock = None
    if lock_path is not None and lock_path.is_file():
        try:
            source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_lock = None
    expected_commit = source_lock.get("source_commit") if isinstance(source_lock, dict) else None
    source_frozen = (
        expected_commit == actual_commit
        and source_lock.get("allowlist_sha256") == file_sha256(allowlist_file)
    ) if isinstance(source_lock, dict) else False
    check(
        "source_commit_frozen",
        source_frozen,
        "ignored deployment source lock must bind HEAD and the allowlist SHA-256",
    )
    dirty = bool(_git("-C", str(root), "status", "--porcelain"))
    check("source_tree_clean", not dirty, "deployment source must be committed and clean")
    check("fp_only", allowlist.get("fp_only") is True, "allowlist must remain FP-only")
    check("cuda_required", allowlist.get("require_cuda") is True, "CUDA fallback is forbidden")

    resolved_hashes = {}
    for candidate_id, candidate in allowlist.get("candidates", {}).items():
        path = root / candidate["config"]
        actual_hash = file_sha256(path) if path.is_file() else None
        resolved_hashes[candidate_id] = actual_hash
        check(
            f"candidate_{candidate_id}_config_hash",
            actual_hash == candidate["sha256"],
            "candidate config must match the allowlisted SHA-256",
        )
        if not path.is_file():
            continue
        config = load_config(path)
        try:
            identity = architecture_identity(config)
            model = build_separator(config)
            count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            valid = identity["architecture_id"] == candidate["architecture"]
            detail = f"architecture={identity['architecture_id']} trainable_parameters={count}"
        except (TypeError, ValueError, RuntimeError) as error:
            valid = False
            detail = f"construction failed: {error}"
        check(f"candidate_{candidate_id}_construction", valid, detail)
        quant = config.get("quant", {})
        train = config.get("train", {})
        distillation = config.get("distillation", {})
        fp_safe = (
            not quant.get("layer_precisions")
            and str(train.get("amp", "off")).lower() == "off"
            and not distillation.get("enabled", False)
        )
        check(f"candidate_{candidate_id}_fp_boundaries", fp_safe, "QAT, AMP, and distillation denied")

    gate = allowlist.get("gate_policy", {})
    gate_path = root / gate.get("path", "")
    check(
        "gate_policy_hash",
        gate_path.is_file() and file_sha256(gate_path) == gate.get("sha256"),
        "gate policy must match the allowlisted SHA-256",
    )
    controller_evidence = allowlist.get("required_evidence", {}).get(
        "controller_fake_matrix", {}
    )
    controller_path_value = controller_evidence.get("path")
    controller_path = root / controller_path_value if controller_path_value else None
    check(
        "controller_fake_matrix_evidence",
        controller_path is not None
        and controller_path.is_file()
        and file_sha256(controller_path) == controller_evidence.get("sha256"),
        "complete fake-controller crash/supervision matrix evidence is required",
    )
    provenance = allowlist.get("provenance", {})
    provenance_path = root / "docs/provenance/SCNET_5d95bf96.md"
    check(
        "scnet_provenance_hash",
        provenance_path.is_file()
        and file_sha256(provenance_path) == provenance.get("scnet_document_sha256"),
        "SCNet provenance document must match the allowlist",
    )

    evidence_valid = False
    evidence_detail = "authorized local CUDA evidence is missing"
    if cuda_evidence_path is not None and Path(cuda_evidence_path).is_file():
        try:
            evidence = json.loads(Path(cuda_evidence_path).read_text(encoding="utf-8"))
            evidence_valid = (
                evidence.get("schema_version") == 1
                and evidence.get("cuda_available") is True
                and evidence.get("source_commit") == expected_commit
                and evidence.get("allowlist_sha256") == file_sha256(allowlist_file)
                and evidence.get("candidate_config_sha256") == resolved_hashes
            )
            evidence_detail = "CUDA evidence hashes and environment binding checked"
        except (OSError, json.JSONDecodeError):
            evidence_detail = "CUDA evidence is corrupt"
    check("authorized_local_cuda_evidence", evidence_valid, evidence_detail)

    return {
        "schema_version": 1,
        "deployment_id": allowlist.get("deployment_id"),
        "source_commit": actual_commit,
        "allowlist_sha256": file_sha256(allowlist_file),
        "matrix_sha256": canonical_json_sha256(checks),
        "ready": all(item["passed"] for item in checks),
        "checks": checks,
    }


def freeze_source(repository: str | Path, allowlist_path: str | Path) -> dict:
    root = Path(repository).resolve()
    if _git("-C", str(root), "status", "--porcelain"):
        raise ValueError("source tree must be clean before freezing deployment source")
    allowlist_file = root / allowlist_path
    allowlist = yaml.safe_load(allowlist_file.read_text(encoding="utf-8"))
    lock_value = allowlist.get("source_commit_file")
    if not isinstance(lock_value, str) or not lock_value:
        raise ValueError("allowlist source_commit_file is missing")
    return {
        "schema_version": 1,
        "source_commit": _git("-C", str(root), "rev-parse", "HEAD"),
        "allowlist_sha256": file_sha256(allowlist_file),
        "source_lock_path": lock_value,
    }


def cuda_probe(repository: str | Path, allowlist_path: str | Path) -> dict:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ValueError("authorized CUDA probe requires a working CUDA device")
    root = Path(repository).resolve()
    allowlist_file = root / allowlist_path
    allowlist = yaml.safe_load(allowlist_file.read_text(encoding="utf-8"))
    lock_path = root / allowlist["source_commit_file"]
    if not lock_path.is_file():
        raise ValueError("deployment source lock is missing")
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        source_lock.get("source_commit") != _git("-C", str(root), "rev-parse", "HEAD")
        or source_lock.get("allowlist_sha256") != file_sha256(allowlist_file)
    ):
        raise ValueError("deployment source lock does not match HEAD and allowlist")
    config_hashes = {
        candidate_id: file_sha256(root / candidate["config"])
        for candidate_id, candidate in allowlist["candidates"].items()
    }
    return {
        "schema_version": 1,
        "source_commit": source_lock["source_commit"],
        "allowlist_sha256": file_sha256(allowlist_file),
        "candidate_config_sha256": config_hashes,
        "cuda_available": True,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
