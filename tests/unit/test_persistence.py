import json

import torch

from ternarystem.training import atomic_json_save, atomic_torch_save, capture_rng_state


def test_atomic_torch_save_publishes_loadable_checkpoint(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_torch_save({"state_dict": {"weight": torch.tensor([1.0])}}, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(payload["state_dict"]["weight"], torch.tensor([1.0]))
    assert not list(tmp_path.glob("*.tmp"))


def test_rng_checkpoint_remains_weights_only_loadable(tmp_path):
    path = tmp_path / "rng.pt"
    atomic_torch_save({"state_dict": {}, "rng_state": capture_rng_state()}, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_atomic_json_save_replaces_existing_record(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    atomic_json_save({"schema_version": 2, "training": []}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "training": [],
    }
    assert not list(tmp_path.glob("*.tmp"))
