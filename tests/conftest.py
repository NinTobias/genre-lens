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
        "training": {
            "cv_folds": 3,
            "output_dir": str(tmp_path / "reports"),
            "model_dir": str(tmp_path / "models"),
        },
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


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Feature table without audio: 4 genres x 15 tracks x 6 windows, separable by design."""
    rng = np.random.default_rng(0)
    genres = ["blues", "jazz", "metal", "pop"]
    n_features = 8
    rows = []
    for g_idx, genre in enumerate(genres):
        genre_center = np.zeros(n_features)
        genre_center[g_idx * 2 : g_idx * 2 + 2] = 3.0
        for t in range(15):
            track_offset = rng.normal(0, 0.5, n_features)
            for w in range(6):
                values = genre_center + track_offset + rng.normal(0, 1.0, n_features)
                rows.append(
                    {"track_id": f"{genre}/{genre}.{t:05d}", "genre": genre, "window": w}
                    | {f"feat_{i}": v for i, v in enumerate(values)}
                )
    return pd.DataFrame(rows)
