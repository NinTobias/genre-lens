"""End-to-end smoke test: discover -> split -> features -> train -> predict."""

from sklearn.ensemble import RandomForestClassifier

from genre_lens.data import discover_tracks, split_tracks, subset_by_tracks
from genre_lens.features import META_COLUMNS, build_feature_table


def test_pipeline_runs_end_to_end(audio_dataset, cfg):
    tracks = discover_tracks(audio_dataset)
    train_tracks, test_tracks = split_tracks(tracks, test_size=0.25, seed=cfg["seed"])
    features = build_feature_table(tracks, cfg)

    train = subset_by_tracks(features, train_tracks)
    test = subset_by_tracks(features, test_tracks)
    assert set(train["track_id"]).isdisjoint(test["track_id"])

    X_cols = [c for c in features.columns if c not in META_COLUMNS]
    model = RandomForestClassifier(n_estimators=20, random_state=cfg["seed"])
    model.fit(train[X_cols], train["genre"])
    predictions = model.predict(test[X_cols])

    assert set(predictions) <= {"low", "high"}
    # 220 Hz vs 880 Hz tones are trivially separable
    assert (predictions == test["genre"]).mean() == 1.0
