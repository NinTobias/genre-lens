"""Dataset discovery and leakage-safe splitting.

The key rule of this module: splits are always made on the *track* level.
Later, every track is cut into several short windows. All windows of one track
must end up on the same side of a split, otherwise the model sees near-identical
audio in training and testing and the reported accuracy is unrealistically high.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

AUDIO_EXTENSIONS = {".wav", ".au", ".mp3", ".flac", ".ogg"}


class LeakageError(ValueError):
    """Raised when the same track appears in two splits."""


def discover_tracks(root: str | Path) -> pd.DataFrame:
    """Scan ``root/<genre>/<file>`` and return one row per track.

    Returns a DataFrame with the columns ``track_id``, ``genre`` and ``path``.
    The result is sorted, so it is deterministic across machines.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. See data/README.md for download instructions."
        )

    rows = []
    for genre_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in sorted(genre_dir.iterdir()):
            if file.suffix.lower() in AUDIO_EXTENSIONS:
                rows.append(
                    {
                        "track_id": f"{genre_dir.name}/{file.stem}",
                        "genre": genre_dir.name,
                        "path": str(file),
                    }
                )

    if not rows:
        raise ValueError(f"No audio files found under {root}.")

    tracks = pd.DataFrame(rows)
    if tracks["track_id"].duplicated().any():
        raise ValueError("Duplicate track ids found (same name with different extensions?).")
    return tracks


def assert_no_leakage(a_ids: Iterable[str], b_ids: Iterable[str]) -> None:
    """Raise :class:`LeakageError` if the two collections of track ids overlap."""
    overlap = set(a_ids) & set(b_ids)
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise LeakageError(f"{len(overlap)} track(s) appear in both splits, e.g. {examples}")


def split_tracks(
    tracks: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the track level."""
    train, test = train_test_split(
        tracks, test_size=test_size, stratify=tracks["genre"], random_state=seed
    )
    train = train.sort_values("track_id").reset_index(drop=True)
    test = test.sort_values("track_id").reset_index(drop=True)
    assert_no_leakage(train["track_id"], test["track_id"])
    return train, test


def subset_by_tracks(features: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Select all window rows that belong to the given tracks."""
    return features[features["track_id"].isin(tracks["track_id"])].reset_index(drop=True)


def group_cv_splits(
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified cross-validation where each track (group) is in exactly one fold.

    Yields ``(train_idx, val_idx)`` index arrays and verifies the absence of leakage.
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in cv.split(np.zeros(len(y)), y, groups=groups):
        assert_no_leakage(groups[train_idx], groups[val_idx])
        yield train_idx, val_idx
