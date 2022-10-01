from pathlib import Path
from countergen.augmentation.data_augmentation import DEFAULT_DS_PATHS, DEFAULT_AUGMENTED_DS_PATHS


def test_ds_path_exist():
    """Paths in DEFAULT_DS_PATHS should exists."""
    for path in DEFAULT_DS_PATHS.values():
        assert Path(path).exists(), path


def test_augmented_ds_path_exist():
    """Paths in DEFAULT_AUGMENTED_DS_PATHS should exists."""
    for path in DEFAULT_AUGMENTED_DS_PATHS.values():
        assert Path(path).exists(), path
