from courtvision.ml.train import train_from_config


def test_heldout_season_is_not_used_to_fit(tmp_path):
    report = train_from_config(
        {
            "fixture_path": "data/fixtures/synthetic_season.json",
            "train_seasons": [2022],
            "validation_seasons": [2023],
            "test_seasons": [2024],
            "min_prior_games": 3,
            "random_seed": 42,
            "evaluate_heldout": False,
            "model_version": "ml-test",
        },
        output_dir=tmp_path,
    )
    assert report["fit_seasons"] == [2022]
    assert 2024 not in report["fit_seasons"]
    assert report["test_used_for_selection"] is False
    assert report["performance_verified"] is False
    assert "brier_score" in report["metrics"]["logistic"]
