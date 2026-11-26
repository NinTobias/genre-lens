"""Shared fixtures: tiny synthetic datasets, so tests need no real audio data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

SR = 22050


def make_tone(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def cfg(tmp_path) -> dict:
    return {
        "seed": 42,
        "audio": {"sample_rate": SR, "clip_seconds": 30, "window_seconds": 3},
        "features": {"n_mfcc": 20, "cache_path": str(tmp_path / "features.parquet")},
        "data": {"path": str(tmp_path / "genres"), "test_size": 0.25},
    }


@pytest.fixture
def audio_dataset(tmp_path):
    """Two 'genres' with 4 tracks each; every track is 6 s long (= 2 windows)."""
    root = tmp_path / "genres"
    for genre, freq in [("low", 220.0), ("high", 880.0)]:
        (root / genre).mkdir(parents=True)
        for i in range(4):
            sf.write(root / genre / f"{genre}.{i:05d}.wav", make_tone(freq, 6.0), SR)
    return root


@pytest.fixture
def track_table() -> pd.DataFrame:
    """Metadata-only table: 3 genres x 10 tracks."""
    rows = [
        {"track_id": f"{g}/{g}.{i:05d}", "genre": g, "path": f"/fake/{g}.{i:05d}.wav"}
        for g in ("a", "b", "c")
        for i in range(10)
    ]
    return pd.DataFrame(rows)
