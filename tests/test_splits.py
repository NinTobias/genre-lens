import numpy as np
import pandas as pd
import pytest

from genre_lens.data import (
    LeakageError,
    assert_no_leakage,
    discover_tracks,
    group_cv_splits,
    split_tracks,
    subset_by_tracks,
)


def test_discover_tracks(audio_dataset):
    tracks = discover_tracks(audio_dataset)
    assert len(tracks) == 8
    assert set(tracks["genre"]) == {"low", "high"}
    assert tracks["track_id"].is_unique
    # deterministic order
    assert tracks["track_id"].tolist() == sorted(tracks["track_id"])


def test_discover_tracks_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_tracks(tmp_path / "does_not_exist")


def test_discover_tracks_empty_folder(tmp_path):
    (tmp_path / "genre_a").mkdir()
    with pytest.raises(ValueError):
        discover_tracks(tmp_path)


def test_split_has_no_overlap_and_covers_all(track_table):
    train, test = split_tracks(track_table, test_size=0.2, seed=0)
    assert set(train["track_id"]).isdisjoint(test["track_id"])
    assert len(train) + len(test) == len(track_table)


def test_split_is_stratified(track_table):
    _, test = split_tracks(track_table, test_size=0.2, seed=0)
    # 10 tracks per genre, 20 % -> exactly 2 per genre in test
    assert test["genre"].value_counts().to_dict() == {"a": 2, "b": 2, "c": 2}


def test_split_is_deterministic(track_table):
    a_train, a_test = split_tracks(track_table, seed=7)
    b_train, b_test = split_tracks(track_table, seed=7)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_assert_no_leakage_detects_overlap():
    with pytest.raises(LeakageError):
        assert_no_leakage(["t1", "t2"], ["t2", "t3"])


def test_windows_of_a_track_never_cross_the_split(track_table):
    """Leakage test: every window row of a track stays on one side of the split."""
    # Simulate a feature table with 10 windows per track
    features = (
        track_table[["track_id", "genre"]].loc[track_table.index.repeat(10)].reset_index(drop=True)
    )
    features["window"] = np.tile(np.arange(10), len(track_table))

    train_tracks, test_tracks = split_tracks(track_table, seed=1)
    train_rows = subset_by_tracks(features, train_tracks)
    test_rows = subset_by_tracks(features, test_tracks)

    assert set(train_rows["track_id"]).isdisjoint(test_rows["track_id"])
    assert len(train_rows) + len(test_rows) == len(features)


def test_group_cv_keeps_tracks_in_one_fold(track_table):
    features = (
        track_table[["track_id", "genre"]].loc[track_table.index.repeat(10)].reset_index(drop=True)
    )

    folds = list(group_cv_splits(features["genre"], features["track_id"], n_splits=5, seed=0))
    assert len(folds) == 5
    seen_in_validation: set[str] = set()
    for train_idx, val_idx in folds:
        train_ids = set(features["track_id"].iloc[train_idx])
        val_ids = set(features["track_id"].iloc[val_idx])
        assert train_ids.isdisjoint(val_ids)
        # each track is validated exactly once across all folds
        assert seen_in_validation.isdisjoint(val_ids)
        seen_in_validation |= val_ids
    assert seen_in_validation == set(features["track_id"])
