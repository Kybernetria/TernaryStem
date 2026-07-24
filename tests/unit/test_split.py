import pytest

from ternarystem.data import load_split, split_hash, validate_development_data_root


def test_frozen_validation_split():
    split = load_split()
    assert split["seed"] == 20250218
    assert len(split["validation"]) == 14
    assert len(set(split["validation"])) == 14
    assert len(split_hash()) == 64


def test_development_root_rejects_non_train_and_present_test_sibling(tmp_path):
    parent = tmp_path / "MUSDB18-HQ"
    train = parent / "train"
    train.mkdir(parents=True)
    assert validate_development_data_root(train) == train.resolve()
    with pytest.raises(ValueError, match="canonical"):
        validate_development_data_root(parent)
    (parent / "test").mkdir()
    with pytest.raises(ValueError, match="must be absent"):
        validate_development_data_root(train)


def test_development_root_rejects_symlink_before_audio_access(tmp_path):
    train = tmp_path / "dataset" / "train"
    train.mkdir(parents=True)
    link_parent = tmp_path / "linked"
    link_parent.mkdir()
    link = link_parent / "train"
    link.symlink_to(train, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        validate_development_data_root(link)
