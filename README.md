# genre-lens

Music genre classification from audio features (MFCC, chroma, spectral, rhythm) with a reproducible ML pipeline.

[![CI](https://github.com/NinTobias/genre-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/NinTobias/genre-lens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)

> **Status:** work in progress. Results and figures will be added as the project progresses.

## Motivation

I built [nebulaviz](https://github.com/NinTobias/nebulaviz), a real-time audio visualizer, and worked with spectral analysis and beat detection there. This project asks the next question: how well can those same signals be used to *predict* what kind of music a track is?

## Results

| Model                | Accuracy | Macro-F1 |
|----------------------|----------|----------|
| Dummy baseline       | TBD      | TBD      |
| Logistic Regression  | TBD      | TBD      |
| Random Forest        | TBD      | TBD      |
| SVM (RBF)            | TBD      | TBD      |

Confusion matrix and error analysis: *coming soon* (`reports/figures/`).

## Approach

1. **Data:** GTZAN (10 genres, 100 clips of 30 s each). See [Dataset notes](#dataset-notes).
2. **Features** (per clip or 3 s window, aggregated as mean and standard deviation):
   - MFCCs (20 coefficients)
   - Spectral centroid, bandwidth, rolloff, contrast
   - Chroma
   - Zero-crossing rate, RMS energy
   - Tempo and onset strength (rhythm)
3. **Models:** Logistic Regression, Random Forest, SVM, compared against a dummy baseline.
4. **Evaluation:** stratified cross-validation, accuracy, macro-F1, confusion matrix, feature importance.
5. **Leakage control:** all windows of one track always stay in the same split.

## Quickstart

```bash
git clone https://github.com/NinTobias/genre-lens.git
cd genre-lens
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Place the dataset as described in [`data/README.md`](data/README.md), then extract and cache the features:

```bash
python -m genre_lens.features --n-jobs -1          # writes data/features.parquet
python -m genre_lens.features --force              # recompute after changing the config
```

Training, evaluation and prediction commands will be documented here once implemented.

## Project structure

```
genre-lens/
├── configs/          # Experiment configuration (YAML)
├── data/             # Datasets (not tracked, see data/README.md)
├── notebooks/        # Exploration and error analysis
├── src/genre_lens/   # Package: data, features, dsp, train, evaluate, predict
├── tests/            # Unit and pipeline tests (pytest)
├── models/           # Trained models (not tracked by default)
└── reports/figures/  # Plots used in this README
```

## Roadmap

- [x] Repository setup
- [x] Data loading and track-aware train/test split (with leakage tests)
- [x] Feature extraction with caching (91 features per 3 s window)
- [ ] EDA notebook
- [ ] Baseline and classical models with cross-validation
- [ ] Confusion matrix, feature importance, error analysis
- [x] Unit tests and CI (GitHub Actions)
- [ ] NumPy STFT implementation, validated against `librosa.stft`
- [ ] Prediction CLI
- [ ] Optional: CNN on mel-spectrograms for comparison

## Dataset notes

GTZAN is a common benchmark but has known problems: duplicated clips, mislabeled tracks and distortions (Sturm, 2013). Results should therefore be interpreted with caution and are meant for comparing methods, not as an absolute measure of real-world performance.

## License

MIT, see [LICENSE](LICENSE).
