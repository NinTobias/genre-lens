import json

import joblib
import numpy as np
import pytest

from genre_lens.train import (
    build_models,
    cross_validate_model,
    feature_columns,
    run_experiment,
)


@pytest.fixture
def fast_models():
    models = build_models(seed=0)
    models["random_forest"].set_params(clf__n_estimators=20)
    return models


def test_feature_columns_exclude_metadata(synthetic_features):
    cols = feature_columns(synthetic_features)
    assert cols == [f"feat_{i}" for i in range(8)]


def test_cross_validation_returns_scores_per_fold(synthetic_features, fast_models):
    X = synthetic_features[feature_columns(synthetic_features)]
    scores = cross_validate_model(
        fast_models["logreg"],
        X,
        synthetic_features["genre"],
        synthetic_features["track_id"],
        n_splits=3,
        seed=0,
    )
    assert len(scores) == 3
    assert scores["track_acc"].between(0, 1).all()


def test_experiment_end_to_end(synthetic_features, fast_models, cfg):
    outcome = run_experiment(cfg, features=synthetic_features, models=fast_models, n_jobs=1)
    results = outcome["results"].set_index("model")

    assert set(results.index) == {"dummy", "logreg", "random_forest", "svm_rbf"}
    # 4 balanced classes: the dummy baseline is at chance level
    assert results.loc["dummy", "test_track_acc"] == pytest.approx(0.25, abs=0.1)
    # real models must clearly beat the baseline on this separable data
    for name in ("logreg", "random_forest", "svm_rbf"):
        assert results.loc[name, "test_track_acc"] > 0.8
    assert outcome["best_model"] != "dummy"
    # 4 genres x 15 tracks, 25 % test
    assert outcome["n_train_tracks"] + outcome["n_test_tracks"] == 60


def test_experiment_writes_expected_outputs(synthetic_features, fast_models, cfg, tmp_path):
    outcome = run_experiment(cfg, features=synthetic_features, models=fast_models)

    reports = tmp_path / "reports"
    for name in ("results.csv", "results.md", "metrics.json", "classification_report.txt"):
        assert (reports / name).exists(), name
    assert (reports / "figures" / "confusion_matrix.png").exists()
    assert (reports / "figures" / "feature_importance.png").exists()

    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["best_model_by_cv"] == outcome["best_model"]
    assert metrics["seed"] == cfg["seed"]

    saved = joblib.load(tmp_path / "models" / "best_model.joblib")
    assert saved["name"] == outcome["best_model"]
    assert saved["feature_columns"] == [f"feat_{i}" for i in range(8)]
    # the saved model can predict from raw feature columns
    X = synthetic_features[saved["feature_columns"]].head(5)
    assert len(saved["model"].predict(X)) == 5


def test_experiment_is_reproducible(synthetic_features, fast_models, cfg):
    a = run_experiment(cfg, features=synthetic_features, models=fast_models, save=False)
    b = run_experiment(cfg, features=synthetic_features, models=fast_models, save=False)
    np.testing.assert_allclose(
        a["results"].drop(columns="model").to_numpy(), b["results"].drop(columns="model").to_numpy()
    )


def test_no_outputs_written_when_save_is_false(synthetic_features, fast_models, cfg, tmp_path):
    run_experiment(cfg, features=synthetic_features, models=fast_models, save=False)
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "models").exists()
