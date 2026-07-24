"""Exact-hash offline readiness checks for GPU deployment #2."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ternarystem.audio import mixture_consistency
from ternarystem.config import load_config
from ternarystem.losses import complex_l1, multiresolution_stft_loss
from ternarystem.models import architecture_identity, build_separator
from ternarystem.training import (
    atomic_torch_save,
    build_scheduler,
    canonical_json_sha256,
    capture_rng_state,
    load_checkpoint,
    resume_training,
)

_CONTROLLER_EVIDENCE_FILES = {
    "src/ternarystem/deployment/controller.py",
    "src/ternarystem/deployment/gates.py",
    "src/ternarystem/deployment/ledger.py",
    "src/ternarystem/deployment/process.py",
    "src/ternarystem/deployment/state.py",
    "src/ternarystem/deployment/sync.py",
    "tests/unit/test_artifact_sync.py",
    "tests/unit/test_deployment_state.py",
    "tests/unit/test_gates.py",
    "tests/unit/test_process_supervision.py",
    "tests/unit/test_serial_controller.py",
}
_CONTROLLER_EVIDENCE_COMMAND = (
    "pytest -q tests/unit/test_serial_controller.py tests/unit/test_artifact_sync.py "
    "tests/unit/test_deployment_state.py tests/unit/test_process_supervision.py "
    "tests/unit/test_gates.py"
)


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
    candidate_requirements = {}
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
            train = config["train"]
            batch_size = int(train["batch_size"])
            candidate_requirements[candidate_id] = {
                "batch_size": batch_size,
                "gradient_accumulation_steps": int(train.get("gradient_accumulation_steps", 1)),
                "samples": round(
                    float(config["data"]["sample_rate"])
                    * float(config["data"]["chunk_seconds"])
                ),
                "trainable_parameters": count,
            }
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
    controller_valid = False
    if (
        controller_path is not None
        and controller_path.is_file()
        and file_sha256(controller_path) == controller_evidence.get("sha256")
    ):
        try:
            controller_payload = json.loads(controller_path.read_text(encoding="utf-8"))
            recorded_hashes = controller_payload.get("file_sha256")
            matrix_result = controller_payload.get("result")
            controller_valid = (
                controller_payload.get("schema_version") == 1
                and controller_payload.get("command") == _CONTROLLER_EVIDENCE_COMMAND
                and isinstance(matrix_result, dict)
                and matrix_result.get("failed") == 0
                and matrix_result.get("passed") == 36
                and isinstance(recorded_hashes, dict)
                and set(recorded_hashes) == _CONTROLLER_EVIDENCE_FILES
                and all(
                    isinstance(relative_path, str)
                    and isinstance(expected_hash, str)
                    and (root / relative_path).is_file()
                    and file_sha256(root / relative_path) == expected_hash
                    for relative_path, expected_hash in recorded_hashes.items()
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            controller_valid = False
    check(
        "controller_fake_matrix_evidence",
        controller_valid,
        "complete current-source fake-controller crash/supervision matrix evidence is required",
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
            if not isinstance(evidence, dict):
                raise TypeError("CUDA evidence root must be a mapping")
            candidate_cuda = evidence.get("candidate_results")
            required_true = (
                "passed",
                "forward",
                "backward",
                "finite_loss",
                "finite_gradients",
                "fp32",
                "no_cpu_fallback",
                "checkpoint_save",
                "checkpoint_load",
                "checkpoint_resume",
                "checkpoint_state_exact",
                "optimizer_state_exact",
                "scheduler_state_exact",
                "rng_state_exact",
            )

            def finite_number(value: object, *, nonnegative: bool = False) -> bool:
                return (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and torch.isfinite(torch.tensor(float(value))).item()
                    and (not nonnegative or value >= 0)
                )

            def candidate_evidence_valid(candidate_id: str, result: object) -> bool:
                required = candidate_requirements.get(candidate_id)
                return (
                    isinstance(result, dict)
                    and isinstance(required, dict)
                    and all(result.get(field) is True for field in required_true)
                    and result.get("batch_size") == required["batch_size"]
                    and result.get("gradient_accumulation_steps")
                    == required["gradient_accumulation_steps"]
                    and result.get("samples") == required["samples"]
                    and result.get("trainable_parameters")
                    == required["trainable_parameters"]
                    and result.get("output_shape")
                    == [required["batch_size"], 4, 2, required["samples"]]
                    and result.get("optimizer_updates") == 1
                    and result.get("resume_start_epoch") == 1
                    and finite_number(result.get("loss"), nonnegative=True)
                    and finite_number(result.get("gradient_norm"), nonnegative=True)
                    and finite_number(
                        result.get("mixture_consistency_max_abs"), nonnegative=True
                    )
                    and isinstance(result.get("peak_allocated_bytes"), int)
                    and not isinstance(result.get("peak_allocated_bytes"), bool)
                    and result["peak_allocated_bytes"] > 0
                    and isinstance(result.get("peak_reserved_bytes"), int)
                    and not isinstance(result.get("peak_reserved_bytes"), bool)
                    and result["peak_reserved_bytes"] >= result["peak_allocated_bytes"]
                )

            evidence_valid = (
                evidence.get("schema_version") == 2
                and evidence.get("cuda_available") is True
                and evidence.get("passed") is True
                and isinstance(evidence.get("torch"), str)
                and bool(evidence["torch"])
                and isinstance(evidence.get("torch_cuda"), str)
                and bool(evidence["torch_cuda"])
                and isinstance(evidence.get("gpu"), str)
                and bool(evidence["gpu"])
                and evidence.get("source_commit") == expected_commit
                and evidence.get("allowlist_sha256") == file_sha256(allowlist_file)
                and evidence.get("candidate_config_sha256") == resolved_hashes
                and isinstance(candidate_cuda, dict)
                and set(candidate_cuda) == {"A", "B"}
                and all(
                    candidate_evidence_valid(candidate_id, candidate_cuda[candidate_id])
                    for candidate_id in ("A", "B")
                )
            )
            evidence_detail = "CUDA evidence hashes and environment binding checked"
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
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


def _probe_optimizer(candidate_id: str, model: torch.nn.Module, train: dict):
    if candidate_id == "A":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(train["learning_rate"]),
            weight_decay=float(train["weight_decay"]),
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=float(train["learning_rate"]),
        betas=tuple(train["betas"]),
        weight_decay=float(train["weight_decay"]),
    )


def _state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _candidate_cuda_probe(root: Path, candidate_id: str, config_path: str) -> dict:
    config = load_config(root / config_path)
    train = config["train"]
    batch_size = int(train["batch_size"])
    samples = round(float(config["data"]["sample_rate"]) * float(config["data"]["chunk_seconds"]))
    device = torch.device("cuda:0")
    accumulation_steps = int(train.get("gradient_accumulation_steps", 1))
    result = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "samples": samples,
    }
    model = optimizer = scheduler = mixture = targets = estimates = loss = None
    try:
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        model = build_separator(config).to(device).train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        result["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        if not parameters or not all(
            parameter.device.type == "cuda" and parameter.dtype == torch.float32
            for parameter in parameters
        ):
            raise RuntimeError("candidate parameters are not entirely FP32 CUDA tensors")
        optimizer = _probe_optimizer(candidate_id, model, train)
        scheduler = build_scheduler(optimizer, train)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        consistency_error = None
        for _ in range(accumulation_steps):
            targets = torch.randn(
                batch_size, 4, 2, samples, device=device, dtype=torch.float32
            ).mul_(0.01)
            mixture = targets.sum(dim=1)
            if candidate_id == "A":
                estimate_spectra = model.spectrograms(mixture)
                estimates = model.stft.synthesis(estimate_spectra, samples)
                target_spectra = model.stft.analysis(targets)
                loss = float(train["waveform_l1"]) * F.l1_loss(
                    estimates, targets
                ) + float(train["complex_l1"]) * complex_l1(
                    estimate_spectra, target_spectra
                )
                adapted = estimates
            else:
                estimates = model.training_estimates(mixture)
                loss = model.training_loss(estimates, targets)
                estimate_spectra = model.stft.analysis(estimates)
                adapted = mixture_consistency(estimates, mixture)
            # Match the production loop's diagnostic tensors, which remain live
            # until backward and materially affect peak CUDA memory.
            diagnostic_target_spectra = model.stft.analysis(targets)
            diagnostic_waveform_loss = F.l1_loss(estimates, targets)
            diagnostic_spectrum_loss = complex_l1(
                estimate_spectra, diagnostic_target_spectra
            )
            if not all(
                bool(torch.isfinite(metric))
                for metric in (diagnostic_waveform_loss, diagnostic_spectrum_loss)
            ):
                raise RuntimeError("candidate diagnostic loss became non-finite")
            if float(train.get("multires_stft", 0)) > 0:
                loss = loss + float(train["multires_stft"]) * multiresolution_stft_loss(
                    estimates, targets
                )
            if (
                estimates.device.type != "cuda"
                or estimates.dtype != torch.float32
                or loss.device.type != "cuda"
                or loss.dtype != torch.float32
                or not bool(torch.isfinite(loss))
            ):
                raise RuntimeError("candidate forward or loss left FP32 CUDA or became non-finite")
            consistency_error = (adapted.sum(dim=1) - mixture).abs().max()
            accumulated_loss += float(loss.detach())
            (loss / accumulation_steps).backward()
        gradients = [parameter.grad for parameter in parameters]
        if not all(
            gradient is not None
            and gradient.device.type == "cuda"
            and gradient.dtype == torch.float32
            and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ):
            raise RuntimeError("candidate gradients are missing, non-finite, or off CUDA FP32")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("candidate gradient norm is non-finite")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        torch.cuda.synchronize(device)
        result.update(
            {
                "forward": True,
                "backward": True,
                "finite_loss": True,
                "loss": accumulated_loss / accumulation_steps,
                "finite_gradients": True,
                "gradient_norm": float(gradient_norm),
                "output_shape": list(estimates.shape),
                "mixture_consistency_max_abs": float(consistency_error),
                "optimizer_updates": 1,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "fp32": True,
                "no_cpu_fallback": True,
            }
        )
        del mixture, targets, estimates, loss, adapted
        mixture = targets = estimates = loss = None
        for parameter in parameters:
            parameter.grad = None
        with tempfile.TemporaryDirectory(prefix=f"deployment2-{candidate_id}-") as directory:
            checkpoint_path = Path(directory) / "resume.pt"
            payload = {
                "checkpoint_schema_version": 2,
                "architecture": architecture_identity(config),
                "resolved_config": config,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": 0,
                "rng_state": capture_rng_state(),
            }
            atomic_torch_save(payload, checkpoint_path)
            expected_cpu_rng = torch.rand(8)
            expected_cuda_rng = torch.rand(8, device=device)
            loaded = load_checkpoint(checkpoint_path, device)
            restored = build_separator(config).to(device)
            restored_optimizer = _probe_optimizer(candidate_id, restored, train)
            restored_scheduler = build_scheduler(restored_optimizer, train)
            next_epoch = resume_training(
                restored, restored_optimizer, loaded, restored_scheduler
            )
            model_exact = all(
                torch.equal(value, restored.state_dict()[key])
                for key, value in model.state_dict().items()
            )
            optimizer_exact = _state_equal(
                optimizer.state_dict(), restored_optimizer.state_dict()
            )
            scheduler_exact = _state_equal(
                scheduler.state_dict() if scheduler is not None else None,
                restored_scheduler.state_dict() if restored_scheduler is not None else None,
            )
            rng_exact = torch.equal(torch.rand(8), expected_cpu_rng) and torch.equal(
                torch.rand(8, device=device), expected_cuda_rng
            )
            if not (
                next_epoch == 1
                and model_exact
                and optimizer_exact
                and scheduler_exact
                and rng_exact
            ):
                raise RuntimeError("strict CUDA checkpoint resume was not exact")
            result.update(
                {
                    "checkpoint_save": True,
                    "checkpoint_load": True,
                    "checkpoint_resume": True,
                    "checkpoint_state_exact": model_exact,
                    "optimizer_state_exact": optimizer_exact,
                    "scheduler_state_exact": scheduler_exact,
                    "rng_state_exact": rng_exact,
                    "resume_start_epoch": next_epoch,
                }
            )
            del loaded, restored, restored_optimizer, restored_scheduler, payload
        result["passed"] = True
    except Exception as error:  # evidence must preserve the actual CUDA failure
        result.update({"passed": False, "error_type": type(error).__name__, "error": str(error)})
        result.setdefault("peak_allocated_bytes", torch.cuda.max_memory_allocated(device))
        result.setdefault("peak_reserved_bytes", torch.cuda.max_memory_reserved(device))
    finally:
        del model, optimizer, scheduler, mixture, targets, estimates, loss
        gc.collect()
        torch.cuda.empty_cache()
    return result


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
        or _git("-C", str(root), "status", "--porcelain")
    ):
        raise ValueError("deployment source lock requires a clean matching HEAD and allowlist")
    config_hashes = {
        candidate_id: file_sha256(root / candidate["config"])
        for candidate_id, candidate in allowlist["candidates"].items()
    }
    candidate_results = {
        candidate_id: _candidate_cuda_probe(root, candidate_id, allowlist["candidates"][candidate_id]["config"])
        for candidate_id in ("A", "B")
    }
    return {
        "schema_version": 2,
        "source_commit": source_lock["source_commit"],
        "allowlist_sha256": file_sha256(allowlist_file),
        "candidate_config_sha256": config_hashes,
        "cuda_available": True,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "candidate_results": candidate_results,
        "passed": all(result["passed"] for result in candidate_results.values()),
    }
