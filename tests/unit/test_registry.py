import copy

import pytest
import torch

from ternarystem.config import model_config
from ternarystem.models import (
    LEGACY_ARCHITECTURE_ID,
    Separator,
    architecture_identity,
    build_from_checkpoint,
    build_separator,
    checkpoint_architecture_id,
    registered_architectures,
)


def tiny_resolved_config(architecture=None):
    model = {
        "channels": [4],
        "tdf_bottleneck": 4,
        "n_fft": 32,
        "hop_length": 8,
        "frequency_bins": 16,
        "sources": 4,
    }
    if architecture is not None:
        model["architecture"] = architecture
    return {"model": model, "quant": {"layer_precisions": {}}}


def test_missing_architecture_narrowly_migrates_to_legacy():
    config = tiny_resolved_config()
    resolved = model_config(config)
    model = build_separator(config)
    assert isinstance(model, Separator)
    assert resolved == model.config
    assert architecture_identity(config)["architecture_id"] == LEGACY_ARCHITECTURE_ID
    assert LEGACY_ARCHITECTURE_ID in registered_architectures()


def test_explicit_legacy_architecture_preserves_state_dict_and_forward():
    old_config = tiny_resolved_config()
    explicit = copy.deepcopy(old_config)
    explicit["model"]["architecture"] = LEGACY_ARCHITECTURE_ID
    torch.manual_seed(5)
    old = build_separator(old_config).eval()
    new = build_separator(explicit).eval()
    new.load_state_dict(copy.deepcopy(old.state_dict()), strict=True)
    mixture = torch.randn(1, 2, 128)
    with torch.inference_mode():
        torch.testing.assert_close(new(mixture), old(mixture))


def test_unknown_architecture_fails_closed():
    with pytest.raises(ValueError, match="unknown model architecture"):
        build_separator(tiny_resolved_config("misspelled_model"))


def test_checkpoint_identity_and_legacy_migration():
    config = tiny_resolved_config()
    model = build_separator(config)
    legacy = {"config": model.config.__dict__, "state_dict": model.state_dict()}
    assert checkpoint_architecture_id(legacy) == LEGACY_ARCHITECTURE_ID
    restored = build_from_checkpoint(legacy)
    restored.load_state_dict(legacy["state_dict"], strict=True)

    identified = dict(legacy, architecture=architecture_identity(config))
    assert checkpoint_architecture_id(identified) == LEGACY_ARCHITECTURE_ID


def test_checkpoint_identity_conflict_is_rejected():
    config = tiny_resolved_config()
    model = build_separator(config)
    payload = {
        "architecture": {
            "architecture_id": "not_registered",
            "schema_version": 1,
            "provenance_sha256": None,
        },
        "config": model.config.__dict__,
        "state_dict": model.state_dict(),
    }
    with pytest.raises(ValueError, match="unknown model architecture"):
        build_from_checkpoint(payload)
