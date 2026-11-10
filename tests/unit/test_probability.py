from courtvision.ml.baselines import ConstantProbability
from courtvision.ml.predict import PredictionService


def test_probability_contract():
    prediction_service = PredictionService(
        games=[],
        known_team_ids={1, 2},
        model=ConstantProbability(0.5),
        min_prior_games=0,
    )
    result = prediction_service.predict(home_id=1, away_id=2, as_of="2024-01-10")
    assert result.p_home_win is not None and result.p_away_win is not None
    assert 0 <= result.p_home_win <= 1
    assert 0 <= result.p_away_win <= 1
    assert abs(result.p_home_win + result.p_away_win - 1) < 1e-8
    assert result.model_version
    assert result.abstained is False


def test_insufficient_history_abstains_without_a_fake_probability():
    service = PredictionService(games=[], known_team_ids={1, 2}, min_prior_games=5)
    result = service.predict(home_id=1, away_id=2, as_of="2024-01-10")
    assert result.abstained is True
    assert result.p_home_win is None
    assert result.p_away_win is None
    assert result.data_quality == "insufficient"
