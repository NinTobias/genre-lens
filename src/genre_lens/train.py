"""Train and compare models.

Protocol
--------
1. Split the *tracks* (not windows) into train and test, stratified by genre.
2. Compare all models with track-grouped cross-validation on the train tracks only.
3. Fit every model on the full train set and evaluate it once on the untouched test tracks.
4. Pick the "best" model by cross-validation score (never by test score), so the test
   set stays an honest final check.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from genre_lens.config import DEFAULT_CONFIG_PATH, load_config
from genre_lens.data import group_cv_splits, split_tracks, subset_by_tracks
from genre_lens.evaluate import (
    aggregate_by_track,
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    to_markdown_table,
    top_confusions,
)
from genre_lens.features import META_COLUMNS, load_or_build_features

logger = logging.getLogger(__name__)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in META_COLUMNS]


def build_models(seed: int = 42, n_jobs: int = 1) -> dict[str, Pipeline]:
    """The models to compare. Every model is a Pipeline so that scaling is fit on train only."""
    return {
        "dummy": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(n_estimators=300, n_jobs=n_jobs, random_state=seed),
                )
            ]
        ),
        "svm_rbf": Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", C=10))]),
    }


def _track_scores(track_ids: Any, genres: Any, pred: Any) -> dict[str, float]:
    frame = pd.DataFrame({"track_id": track_ids, "genre": genres, "pred": pred})
    tracks = aggregate_by_track(frame)
    return compute_metrics(tracks["genre"], tracks["pred"])


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Track-grouped CV. Returns one row of scores per fold."""
    rows = []
    for train_idx, val_idx in group_cv_splits(y, groups, n_splits=n_splits, seed=seed):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[val_idx])
        window = compute_metrics(y.iloc[val_idx], pred)
        track = _track_scores(groups.iloc[val_idx], y.iloc[val_idx], pred)
        rows.append(
            {
                "window_acc": window["accuracy"],
                "track_acc": track["accuracy"],
                "track_macro_f1": track["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    cfg: dict[str, Any],
    features: pd.DataFrame | None = None,
    models: dict[str, Pipeline] | None = None,
    n_jobs: int = 1,
    save: bool = True,
) -> dict[str, Any]:
    """Run the full comparison and (optionally) write results, figures and the best model."""
    seed = cfg["seed"]
    tcfg = cfg.get("training", {})
    cv_folds = tcfg.get("cv_folds", 5)
    report_dir = Path(tcfg.get("output_dir", "reports"))
    model_dir = Path(tcfg.get("model_dir", "models"))

    if features is None:
        features = load_or_build_features(cfg, n_jobs=n_jobs)
    models = models if models is not None else build_models(seed, n_jobs=n_jobs)

    tracks = features[["track_id", "genre"]].drop_duplicates("track_id").reset_index(drop=True)
    train_tracks, test_tracks = split_tracks(tracks, cfg["data"]["test_size"], seed)
    train = subset_by_tracks(features, train_tracks)
    test = subset_by_tracks(features, test_tracks)
    cols = feature_columns(features)
    X_train, y_train, groups_train = train[cols], train["genre"], train["track_id"]
    X_test, y_test = test[cols], test["genre"]
    logger.info(
        "Train: %d tracks / %d windows. Test: %d tracks / %d windows. %d features.",
        len(train_tracks),
        len(train),
        len(test_tracks),
        len(test),
        len(cols),
    )

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    test_track_preds: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        logger.info("Cross-validating %s ...", name)
        cv = cross_validate_model(model, X_train, y_train, groups_train, cv_folds, seed)
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_test)
        window = compute_metrics(y_test, pred)
        track_frame = aggregate_by_track(
            pd.DataFrame({"track_id": test["track_id"], "genre": y_test, "pred": pred})
        )
        track = compute_metrics(track_frame["genre"], track_frame["pred"])
        rows.append(
            {
                "model": name,
                "cv_track_acc": float(cv["track_acc"].mean()),
                "cv_track_acc_std": float(cv["track_acc"].std(ddof=0)),
                "test_track_acc": track["accuracy"],
                "test_track_macro_f1": track["macro_f1"],
                "test_window_acc": window["accuracy"],
            }
        )
        fitted_models[name] = fitted
        test_track_preds[name] = track_frame
        logger.info(
            "  %s: CV track acc %.3f | test track acc %.3f",
            name,
            rows[-1]["cv_track_acc"],
            track["accuracy"],
        )

    results = pd.DataFrame(rows)
    candidates = results[results["model"] != "dummy"]
    pool = candidates if len(candidates) else results
    best_name = str(pool.sort_values("cv_track_acc", ascending=False).iloc[0]["model"])
    best_tracks = test_track_preds[best_name]
    confusions = top_confusions(best_tracks["genre"], best_tracks["pred"], k=5)

    outcome = {
        "results": results,
        "best_model": best_name,
        "top_confusions": confusions,
        "n_train_tracks": len(train_tracks),
        "n_test_tracks": len(test_tracks),
    }

    if save:
        _save_outputs(
            outcome, fitted_models, best_tracks, cols, report_dir, model_dir, seed, cfg["data"]
        )
    return outcome


def _save_outputs(
    outcome: dict[str, Any],
    fitted_models: dict[str, Pipeline],
    best_tracks: pd.DataFrame,
    cols: list[str],
    report_dir: Path,
    model_dir: Path,
    seed: int,
    data_cfg: dict[str, Any],
) -> None:
    results: pd.DataFrame = outcome["results"]
    best_name: str = outcome["best_model"]
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(report_dir / "results.csv", index=False)
    (report_dir / "results.md").write_text(to_markdown_table(results) + "\n", encoding="utf-8")
    (report_dir / "classification_report.txt").write_text(
        f"Model: {best_name} (track level, held-out test tracks)\n\n"
        + classification_report(best_tracks["genre"], best_tracks["pred"], zero_division=0),
        encoding="utf-8",
    )
    metrics = {
        "seed": seed,
        "test_size": data_cfg["test_size"],
        "n_train_tracks": outcome["n_train_tracks"],
        "n_test_tracks": outcome["n_test_tracks"],
        "best_model_by_cv": best_name,
        "top_confusions_best_model": outcome["top_confusions"],
        "results": results.to_dict("records"),
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrix(
        best_tracks["genre"],
        best_tracks["pred"],
        report_dir / "figures" / "confusion_matrix.png",
        title=f"Confusion matrix: {best_name} (track level, test set)",
    )
    forest = fitted_models.get("random_forest")
    if forest is not None:
        plot_feature_importance(
            forest[-1].feature_importances_, cols, report_dir / "figures" / "feature_importance.png"
        )

    joblib.dump(
        {"model": fitted_models[best_name], "feature_columns": cols, "name": best_name},
        model_dir / "best_model.joblib",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare genre classifiers.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 = all cores).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    outcome = run_experiment(load_config(args.config), n_jobs=args.n_jobs)

    print("\n" + to_markdown_table(outcome["results"]))
    print(f"\nBest model by cross-validation: {outcome['best_model']}")
    print("Most frequent confusions (track level, test set):")
    for c in outcome["top_confusions"]:
        print(f"  {c['true']} -> {c['predicted']}: {c['count']}")
    print("\nOutputs written to the reports/ and models/ folders.")


if __name__ == "__main__":
    main()
