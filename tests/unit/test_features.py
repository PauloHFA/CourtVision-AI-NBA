from datetime import date, datetime, timezone

import pytest

from courtvision.data.records import GameRecord
from courtvision.errors import InvalidAsOfError
from courtvision.features.builder import build_matchup_features, elo_before, parse_as_of
from courtvision.features.schema import INITIAL_ELO, NEUTRAL_POINTS, NEUTRAL_REST_DAYS, NEUTRAL_WIN_RATE


def _game(
    game_id: int,
    day: date,
    home: int,
    away: int,
    home_score: int,
    away_score: int,
) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        game_date=day,
        home_team_id=home,
        visitor_team_id=away,
        home_score=home_score,
        visitor_score=away_score,
        status="final",
        season=2023,
        source_game_id=game_id,
    )


def _sample() -> list[GameRecord]:
    return [
        _game(1, date(2024, 1, 1), 1, 2, 110, 100),
        _game(2, date(2024, 1, 3), 2, 1, 100, 90),
        _game(3, date(2024, 1, 5), 1, 2, 120, 100),
        _game(4, date(2024, 1, 8), 1, 2, 100, 110),
        _game(5, date(2024, 1, 10), 1, 2, 105, 100),
        _game(6, date(2024, 1, 12), 1, 2, 130, 90),
    ]


def test_parse_as_of_accepts_date_and_rejects_blank():
    parsed = parse_as_of(date(2024, 4, 1))
    assert parsed == datetime(2024, 4, 1, tzinfo=timezone.utc)
    naive = parse_as_of(datetime(2024, 4, 1, 18, 0))
    assert naive.tzinfo == timezone.utc
    assert parse_as_of("2024-04-01T18:00:00Z").hour == 18
    with pytest.raises(InvalidAsOfError):
        parse_as_of("   ")
    with pytest.raises(InvalidAsOfError):
        parse_as_of("ontem")


def test_rolling_window_uses_only_the_newest_prior_games():
    features = build_matchup_features(_sample(), 1, 2, "2024-01-13T18:00:00Z")
    assert features["home_win_rate_last_5"] == 0.6
    assert features["away_win_rate_last_5"] == 0.4
    assert features["home_points_for_avg_5"] == 109.0
    assert features["home_points_against_avg_5"] == 100.0
    assert features["home_win_rate_at_home"] == 0.8
    assert features["away_win_rate_on_road"] == 0.2
    assert features["home_prior_games"] == 6
    assert features["home_short_history"] == 1.0
    assert features["home_imputed"] == 0.0


def test_rest_counts_days_since_the_last_prior_game():
    next_day = build_matchup_features(_sample(), 1, 2, "2024-01-13")
    assert next_day["home_days_since_last_game"] == 1.0
    assert next_day["home_back_to_back"] == 1.0
    later = build_matchup_features(_sample(), 1, 2, "2024-01-15")
    assert later["home_days_since_last_game"] == 3.0
    assert later["home_back_to_back"] == 0.0


def test_empty_history_is_imputed_and_marked():
    features = build_matchup_features(_sample(), 1, 2, "2023-10-01")
    assert features["home_imputed"] == 1.0
    assert features["away_imputed"] == 1.0
    assert features["home_win_rate_last_5"] == NEUTRAL_WIN_RATE
    assert features["home_points_for_avg_5"] == NEUTRAL_POINTS
    assert features["home_days_since_last_game"] == NEUTRAL_REST_DAYS
    assert features["min_prior_games"] == 0.0
    assert features["home_short_history"] == 1.0


def test_one_prior_game_keeps_the_observed_rate_and_flags_short_history():
    features = build_matchup_features(_sample()[:1], 1, 2, "2024-01-02")
    assert features["home_imputed"] == 0.0
    assert features["home_win_rate_last_5"] == 1.0
    assert features["home_short_history"] == 1.0
    assert features["home_prior_games"] == 1


def test_elo_rewards_the_winner_and_ignores_a_later_game():
    played = _sample()[:1]
    ratings = elo_before(played, date(2024, 1, 2))
    assert ratings[1] > INITIAL_ELO
    assert ratings[2] < INITIAL_ELO
    assert abs((ratings[1] + ratings[2]) - (2 * INITIAL_ELO)) < 1e-9
    later = _game(99, date(2024, 2, 1), 2, 1, 140, 80)
    assert elo_before([*played, later], date(2024, 1, 2)) == ratings
