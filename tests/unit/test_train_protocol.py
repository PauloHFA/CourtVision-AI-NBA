from datetime import date

from courtvision.ml.candidates import xgboost_importable
from courtvision.ml.protocol import expanding_windows, later_window
from courtvision.ml.reliability import reliability_bins, reliability_svg
from courtvision.ml.train import train_from_config


def _rows(count: int, season: int = 2022):
    return [
        {
            "game_id": index,
            "game_date": date(season, 1, index),
            "season": season,
            "y": index % 2,
        }
        for index in range(1, count + 1)
    ]


def test_later_window_is_chronological_and_disjoint():
    early, late = later_window(_rows(4, 2023), 0.5)
    assert [row["game_id"] for row in early] == [1, 2]
    assert [row["game_id"] for row in late] == [3, 4]
    assert {row["game_id"] for row in early}.isdisjoint(row["game_id"] for row in late)


def test_expanding_windows_do_not_look_ahead():
    windows = expanding_windows(_rows(20), min_train=8, min_score=4)
    assert windows
    for train_part, score_part in windows:
        assert max(row["game_date"] for row in train_part) < min(row["game_date"] for row in score_part)
        assert {row["game_id"] for row in train_part}.isdisjoint(row["game_id"] for row in score_part)


def test_reliability_bins_cover_the_sample():
    bins = reliability_bins([1, 0, 1, 0], [0.2, 0.4, 0.8, 0.9], bins=5)
    assert sum(item["count"] for item in bins) == 4
    chart = reliability_svg(bins)
    assert "Não é resultado externo verificado" in chart
    assert "<svg" in chart


def test_training_protocol_keeps_test_and_calibration_out_of_selection(tmp_path):
    report = train_from_config(
        {
            "fixture_path": "data/fixtures/synthetic_season.json",
            "train_seasons": [2022],
            "validation_seasons": [2023],
            "test_seasons": [2024],
            "min_prior_games": 3,
            "random_seed": 42,
            "evaluate_heldout": False,
            "enable_xgboost": True,
            "calibration_method": "sigmoid",
            "calibration_fraction": 0.4,
            "material_brier_gain": 0.005,
            "model_version": "ml-protocol",
        },
        output_dir=tmp_path,
    )
    assert report["fit_seasons"] == [2022]
    assert report["test_used_for_selection"] is False
    assert report["performance_verified"] is False
    assert "heldout_metrics" not in report
    assert report["calibration"]["used_for_selection"] is False
    assert report["calibration"]["overlap_with_fit"] == 0
    assert 2024 not in report["calibration"]["seasons"]
    assert report["walk_forward"]["used_for_selection"] is False
    search = report["walk_forward"]["xgboost_search"]
    assert search["used_validation"] is False
    assert report["material_gain_status"] == "proposed_gate_not_a_measured_result"
    if report["calibration"]["applied"]:
        assert report["calibration"]["fraction"] == 0.4
        assert report["calibration"]["seasons"] == [2023]
        assert report["calibration"]["disjoint_from_fit"] is True
    if xgboost_importable():
        assert search["seasons"] == [2022]
        assert "brier_score" in report["metrics"]["xgboost"]
    else:
        assert report["metrics"]["xgboost"]["status"] == "skipped"
    assert (tmp_path / "reliability_validation.svg").is_file()
    assert "Não é resultado externo verificado" in (tmp_path / "experiment_latest.md").read_text(encoding="utf-8")
