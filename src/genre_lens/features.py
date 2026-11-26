"""Audio feature extraction.

Every track is loaded, cut into fixed-length windows and each window is turned
into one row of summary statistics (mean/std over time) of common audio
descriptors: MFCC, chroma, spectral shape, zero-crossing rate, RMS energy and
rhythm (tempo, onset strength).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from genre_lens.config import DEFAULT_CONFIG_PATH, load_config
from genre_lens.data import discover_tracks

logger = logging.getLogger(__name__)

META_COLUMNS = ["track_id", "genre", "window"]


def load_audio(path: str | Path, sr: int, duration: float | None = None) -> np.ndarray:
    """Load an audio file as mono float32 at the given sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y


def split_windows(y: np.ndarray, sr: int, window_seconds: float) -> list[np.ndarray]:
    """Cut a signal into consecutive windows; an incomplete last window is dropped."""
    size = int(sr * window_seconds)
    if size <= 0:
        raise ValueError("window_seconds must be positive")
    return [y[i * size : (i + 1) * size] for i in range(len(y) // size)]


def zero_crossing_rate(
    y: np.ndarray, frame_length: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Frame-wise zero-crossing rate, shape ``(1, n_frames)``.

    Pure NumPy re-implementation of ``librosa.feature.zero_crossing_rate``. librosa's version
    runs through a numba-compiled function, which crashed on some Windows setups; this one has
    no compiled dependency and is tested to match librosa's output.
    """
    padded = np.pad(y, frame_length // 2, mode="edge")
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    frames = np.where(np.abs(frames) <= 1e-10, 0.0, frames)
    crossings = np.diff(np.signbit(frames), axis=0)
    # librosa pads the first sample of each frame as 'no crossing', i.e. divides by frame_length
    return crossings.sum(axis=0, keepdims=True) / frame_length


def _add_stats(out: dict[str, float], name: str, matrix: np.ndarray) -> None:
    """Add mean and std over time for each row of a (n_rows, n_frames) matrix."""
    mean, std = matrix.mean(axis=1), matrix.std(axis=1)
    single = matrix.shape[0] == 1
    for i, (m, s) in enumerate(zip(mean, std, strict=True), start=1):
        key = name if single else f"{name}_{i:02d}"
        out[f"{key}_mean"] = float(m)
        out[f"{key}_std"] = float(s)


def extract_window_features(y: np.ndarray, sr: int, n_mfcc: int = 20) -> dict[str, float]:
    """Compute the feature vector of a single audio window as an ordered dict."""
    magnitude = np.abs(librosa.stft(y))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    out: dict[str, float] = {}
    # tuning=0.0 skips librosa's automatic tuning estimation (pitch tracking), which is noisy
    # on 3 s windows, slow, and crashed natively on some Windows installs.
    _add_stats(out, "mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc))
    _add_stats(out, "chroma", librosa.feature.chroma_stft(S=magnitude, sr=sr, tuning=0.0))
    _add_stats(out, "spectral_centroid", librosa.feature.spectral_centroid(S=magnitude, sr=sr))
    _add_stats(out, "spectral_bandwidth", librosa.feature.spectral_bandwidth(S=magnitude, sr=sr))
    _add_stats(out, "spectral_rolloff", librosa.feature.spectral_rolloff(S=magnitude, sr=sr))
    _add_stats(out, "spectral_contrast", librosa.feature.spectral_contrast(S=magnitude, sr=sr))
    _add_stats(out, "zcr", zero_crossing_rate(y))
    _add_stats(out, "rms", librosa.feature.rms(S=magnitude))
    _add_stats(out, "onset_strength", onset_env[np.newaxis, :])
    out["tempo"] = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr))[0])
    return out


def extract_track_features(
    path: str | Path,
    sr: int = 22050,
    clip_seconds: float = 30,
    window_seconds: float = 3,
    n_mfcc: int = 20,
) -> list[dict[str, float]]:
    """Return one feature dict per complete window of a track."""
    y = load_audio(path, sr=sr, duration=clip_seconds)
    rows = []
    for index, window in enumerate(split_windows(y, sr, window_seconds)):
        features = extract_window_features(window, sr, n_mfcc=n_mfcc)
        rows.append({"window": index, **features})
    return rows


def _process_track(row: dict[str, Any], audio_cfg: dict, n_mfcc: int) -> list[dict] | None:
    """Worker for one track. Returns None if the file cannot be processed."""
    try:
        windows = extract_track_features(
            row["path"],
            sr=audio_cfg["sample_rate"],
            clip_seconds=audio_cfg["clip_seconds"],
            window_seconds=audio_cfg["window_seconds"],
            n_mfcc=n_mfcc,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught  # corrupt files exist
        logger.warning("Skipping %s: %s", row["track_id"], exc)
        return None
    return [{"track_id": row["track_id"], "genre": row["genre"], **w} for w in windows]


def build_feature_table(tracks: pd.DataFrame, cfg: dict[str, Any], n_jobs: int = 1) -> pd.DataFrame:
    """Extract features for all tracks. Unreadable tracks are skipped with a warning."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_track)(row, cfg["audio"], cfg["features"]["n_mfcc"])
        for row in tracks.to_dict("records")
    )
    rows = [r for track_rows in results if track_rows for r in track_rows]
    if not rows:
        raise ValueError("No features could be extracted from any track.")

    table = pd.DataFrame(rows)
    finite = np.isfinite(table.drop(columns=["track_id", "genre"]).to_numpy()).all(axis=1)
    if not finite.all():
        logger.warning("Dropping %d window(s) with non-finite values.", int((~finite).sum()))
        table = table[finite].reset_index(drop=True)
    return table


def load_or_build_features(
    cfg: dict[str, Any], force: bool = False, n_jobs: int = 1
) -> pd.DataFrame:
    """Load cached features, or compute and cache them.

    The cache is not invalidated automatically when the config changes.
    Use ``force=True`` after changing audio or feature settings.
    """
    cache_path = Path(cfg["features"]["cache_path"])
    if cache_path.exists() and not force:
        logger.info("Loading cached features from %s", cache_path)
        return pd.read_parquet(cache_path)

    tracks = discover_tracks(cfg["data"]["path"])
    table = build_feature_table(tracks, cfg, n_jobs=n_jobs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_path, index=False)
    logger.info("Wrote %d rows to %s", len(table), cache_path)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features and cache them.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--force", action="store_true", help="Recompute even if a cache exists.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (-1 = all cores).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    table = load_or_build_features(load_config(args.config), force=args.force, n_jobs=args.n_jobs)
    print(f"{len(table)} windows, {table['track_id'].nunique()} tracks, {table.shape[1]} columns")


if __name__ == "__main__":
    main()
