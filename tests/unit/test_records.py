from ternarystem.evaluation import base_record


def test_record_captures_explicit_software_and_hardware_metadata():
    record = base_record({"model": {}}, 7, "cpu")
    assert record["schema_version"] == 3
    assert record["host"]["device"] == "cpu"
    assert isinstance(record["host"]["cuda_available"], bool)
    assert "gpu_model" in record["host"]
    assert record["software"]["pytorch"]
    assert "cuda" in record["software"]
    assert record["checkpoint_hashes"] == {"latest": None, "best": None}
