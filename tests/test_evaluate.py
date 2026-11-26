import pandas as pd

from genre_lens.evaluate import (
    aggregate_by_track,
    compute_metrics,
    majority_vote,
    plot_confusion_matrix,
    plot_feature_importance,
    to_markdown_table,
    top_confusions,
)


def test_majority_vote_picks_most_frequent():
    assert majority_vote(pd.Series(["a", "b", "b", "c"])) == "b"


def test_majority_vote_breaks_ties_alphabetically():
    assert majority_vote(pd.Series(["pop", "jazz", "pop", "jazz"])) == "jazz"


def test_aggregate_by_track_gives_one_row_per_track():
    frame = pd.DataFrame(
        {
            "track_id": ["t1"] * 3 + ["t2"] * 3,
            "genre": ["jazz"] * 3 + ["pop"] * 3,
            "pred": ["jazz", "jazz", "pop", "pop", "pop", "jazz"],
        }
    )
    tracks = aggregate_by_track(frame)
    assert tracks["track_id"].tolist() == ["t1", "t2"]
    assert tracks["pred"].tolist() == ["jazz", "pop"]
    assert tracks["genre"].tolist() == ["jazz", "pop"]


def test_compute_metrics_perfect_and_wrong():
    assert compute_metrics(["a", "b"], ["a", "b"]) == {"accuracy": 1.0, "macro_f1": 1.0}
    assert compute_metrics(["a", "b"], ["b", "a"])["accuracy"] == 0.0


def test_top_confusions_sorted_by_count_and_ignores_correct():
    y_true = ["rock", "rock", "rock", "pop", "pop", "jazz"]
    y_pred = ["rock", "metal", "metal", "pop", "rock", "jazz"]
    result = top_confusions(y_true, y_pred, k=5)
    assert result[0] == {"true": "rock", "predicted": "metal", "count": 2}
    assert all(c["true"] != c["predicted"] for c in result)


def test_plots_are_written(tmp_path):
    cm_path = tmp_path / "figs" / "cm.png"
    plot_confusion_matrix(["a", "b", "a", "b"], ["a", "b", "b", "b"], cm_path)
    assert cm_path.exists() and cm_path.stat().st_size > 0

    fi_path = tmp_path / "figs" / "fi.png"
    plot_feature_importance(pd.Series([0.1, 0.5, 0.4]).to_numpy(), ["x", "y", "z"], fi_path)
    assert fi_path.exists() and fi_path.stat().st_size > 0


def test_markdown_table_format():
    df = pd.DataFrame({"model": ["a"], "acc": [0.12345]})
    assert to_markdown_table(df).splitlines() == [
        "| model | acc |",
        "|---|---|",
        "| a | 0.123 |",
    ]
