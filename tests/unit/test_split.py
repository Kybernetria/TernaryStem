from ternarystem.data import load_split, split_hash


def test_frozen_validation_split():
    split = load_split()
    assert split["seed"] == 20250218
    assert len(split["validation"]) == 14
    assert len(set(split["validation"])) == 14
    assert len(split_hash()) == 64
