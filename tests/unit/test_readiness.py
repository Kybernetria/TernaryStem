from pathlib import Path

from ternarystem.deployment.readiness import readiness_matrix

ROOT = Path(__file__).parents[2]


def test_offline_readiness_validates_local_evidence_and_still_requires_cuda():
    result = readiness_matrix(ROOT, "configs/deployment_2/allowlist.yaml")
    by_name = {item["name"]: item for item in result["checks"]}
    assert not result["ready"]
    assert "source_commit_frozen" in by_name
    assert by_name["controller_fake_matrix_evidence"]["passed"]
    assert not by_name["authorized_local_cuda_evidence"]["passed"]
    assert by_name["candidate_C_config_hash"]["passed"]
    assert by_name["candidate_A_config_hash"]["passed"]
    assert by_name["candidate_B_config_hash"]["passed"]
    assert by_name["gate_policy_hash"]["passed"]
    assert by_name["scnet_provenance_hash"]["passed"]
    assert all(by_name[f"candidate_{candidate}_fp_boundaries"]["passed"] for candidate in "CAB")
