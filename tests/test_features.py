import math

import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from genre_lens.data import discover_tracks
from genre_lens.features import (
    build_feature_table,
    extract_track_features,
    extract_window_features,
    load_or_build_features,
    split_windows,
    zero_crossing_rate,
)

from .conftest import SR, make_tone


def test_split_windows_drops_incomplete_tail():
    y = np.zeros(int(SR * 7.5), dtype=np.float32)
    windows = split_windows(y, SR, window_seconds=3)
    assert len(windows) == 2
    assert all(len(w) == 3 * SR for w in windows)


def test_split_windows_rejects_non_positive_window():
    with pytest.raises(ValueError):
        split_windows(np.zeros(100), SR, window_seconds=0)


def test_window_features_are_finite_and_have_stable_keys():
    f1 = extract_window_features(make_tone(440, 3), SR)
    f2 = extract_window_features(make_tone(880, 3), SR)
    assert list(f1) == list(f2)
    assert all(math.isfinite(v) for v in f1.values())


def test_n_mfcc_controls_feature_count():
    small = extract_window_features(make_tone(440, 3), SR, n_mfcc=5)
    large = extract_window_features(make_tone(440, 3), SR, n_mfcc=20)
    # each MFCC contributes mean + std
    assert len(large) - len(small) == (20 - 5) * 2


def test_spectral_centroid_of_pure_tone_is_near_its_frequency():
    f = extract_window_features(make_tone(440, 3), SR)
    assert f["spectral_centroid_mean"] == pytest.approx(440, abs=40)


def test_zero_crossing_rate_matches_theory_for_sine():
    f = extract_window_features(make_tone(440, 3), SR)
    assert f["zcr_mean"] == pytest.approx(2 * 440 / SR, rel=0.05)


def test_higher_tone_has_higher_centroid():
    low = extract_window_features(make_tone(220, 3), SR)
    high = extract_window_features(make_tone(880, 3), SR)
    assert high["spectral_centroid_mean"] > low["spectral_centroid_mean"]


def test_extract_track_features_returns_one_row_per_window(audio_dataset):
    path = next(audio_dataset.rglob("*.wav"))
    rows = extract_track_features(path, sr=SR, clip_seconds=30, window_seconds=3)
    assert [r["window"] for r in rows] == [0, 1]  # 6 s clip -> 2 windows


def test_build_feature_table_has_meta_columns_and_all_tracks(audio_dataset, cfg):
    tracks = discover_tracks(audio_dataset)
    table = build_feature_table(tracks, cfg)
    assert {"track_id", "genre", "window"} <= set(table.columns)
    assert table["track_id"].nunique() == 8
    assert len(table) == 8 * 2
    assert not table.isna().any().any()


def test_corrupt_file_is_skipped(audio_dataset, cfg):
    (audio_dataset / "low" / "low.99999.wav").write_bytes(b"this is not audio")
    tracks = discover_tracks(audio_dataset)
    table = build_feature_table(tracks, cfg)
    assert "low/low.99999" not in set(table["track_id"])
    assert table["track_id"].nunique() == 8


def test_all_files_corrupt_raises(tmp_path, cfg):
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "bad.wav").write_bytes(b"nope")
    with pytest.raises(ValueError):
        build_feature_table(discover_tracks(tmp_path), cfg)


def test_cache_is_written_and_reused(audio_dataset, cfg, tmp_path):
    first = load_or_build_features(cfg)
    assert (tmp_path / "features.parquet").exists()

    # Remove the audio: a second call must succeed purely from the cache.
    for wav in audio_dataset.rglob("*.wav"):
        wav.unlink()
    second = load_or_build_features(cfg)
    pd.testing.assert_frame_equal(first, second)


def test_feature_extraction_is_deterministic(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, make_tone(330, 6.0), SR)
    a = pd.DataFrame(extract_track_features(path, sr=SR))
    b = pd.DataFrame(extract_track_features(path, sr=SR))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numpy_zero_crossing_rate_matches_librosa(seed):
    rng = np.random.default_rng(seed)
    y = (0.3 * make_tone(200, 3) + 0.1 * rng.standard_normal(3 * SR)).astype(np.float32)
    np.testing.assert_allclose(
        zero_crossing_rate(y), librosa.feature.zero_crossing_rate(y), atol=1e-9
    )


def test_chroma_of_a440_peaks_at_pitch_class_a():
    f = extract_window_features(make_tone(440, 3), SR)
    chroma_means = [f[f"chroma_{i:02d}_mean"] for i in range(1, 13)]
    # pitch classes start at C, so A is index 9
    assert int(np.argmax(chroma_means)) == 9
