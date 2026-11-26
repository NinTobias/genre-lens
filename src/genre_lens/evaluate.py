"""Metrics, track-level aggregation and plots.

Models predict one genre per 3 s *window*. A track has about 10 windows, so the
window predictions are combined by majority vote into one prediction per track.
Track-level scores are the ones that matter (and are comparable to the literature),
window-level scores are reported as additional detail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score


def majority_vote(predictions: pd.Series) -> str:
    """Most frequent label; ties are broken alphabetically (deterministic)."""
    counts = predictions.value_counts().sort_index()
    return str(counts.idxmax())


def aggregate_by_track(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine window predictions into one prediction per track.

    ``frame`` needs the columns ``track_id``, ``genre`` (true label) and ``pred``.
    """
    return (
        frame.groupby("track_id")
        .agg(genre=("genre", "first"), pred=("pred", majority_vote))
        .reset_index()
    )


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def top_confusions(y_true: Any, y_pred: Any, k: int = 5) -> list[dict[str, Any]]:
    """The ``k`` most frequent wrong (true -> predicted) genre pairs."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [
        {"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])}
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: (-p["count"], p["true"], p["predicted"]))
    return pairs[:k]


def plot_confusion_matrix(
    y_true: Any, y_pred: Any, path: str | Path, title: str = "Confusion matrix"
) -> None:
    """Save a row-normalised confusion matrix (share of each true genre) as PNG."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f", colorbar=False
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    fig.tight_layout()
    _save(fig, path)


def plot_feature_importance(
    importances: np.ndarray, feature_names: list[str], path: str | Path, top_n: int = 20
) -> None:
    """Save a horizontal bar chart of the ``top_n`` most important features."""
    order = np.argsort(importances)[::-1][:top_n][::-1]
    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    ax.barh([feature_names[i] for i in order], importances[order], color="#4C72B0")
    ax.set_title(f"Top {len(order)} feature importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    _save(fig, path)


def _save(fig: Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def to_markdown_table(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""

    def fmt(value: Any) -> str:
        return float_format.format(value) if isinstance(value, float) else str(value)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])
