from dataclasses import replace
from datetime import date

from courtvision.data.records import GameRecord, dataset_hash
from courtvision.features.builder import build_features


def make_tiny_fixture_with_future_game() -> list[GameRecord]:
    return [
        GameRecord(
            game_id=1,
            game_date=date(2024, 1, 1),
            home_team_id=1,
            visitor_team_id=2,
            home_score=100,
            visitor_score=90,
            status="final",
            season=2023,
            source_game_id=1,
        ),
        GameRecord(
            game_id=10,
            game_date=date(2024, 1, 12),
            home_team_id=1,
            visitor_team_id=2,
            home_score=None,
            visitor_score=None,
            status="scheduled",
            season=2023,
            source_game_id=10,
        ),
        GameRecord(
            game_id=11,
            game_date=date(2024, 1, 20),
            home_team_id=2,
            visitor_team_id=1,
            home_score=80,
            visitor_score=70,
            status="final",
            season=2023,
            source_game_id=11,
        ),
    ]


def edit_score_after_cutoff(games: list[GameRecord], game_id: int, score: int) -> list[GameRecord]:
    updated = []
    for game in games:
        if game.game_id == game_id:
            updated.append(replace(game, home_score=score, visitor_score=score - 1))
        else:
            updated.append(game)
    return updated


def test_future_result_does_not_change_pregame_features():
    games = make_tiny_fixture_with_future_game()
    before = build_features(games, game_id=10, as_of="2024-01-10")
    games = edit_score_after_cutoff(games, game_id=11, score=999)
    after = build_features(games, game_id=10, as_of="2024-01-10")
    assert before == after


def test_editing_a_prior_game_does_change_features_and_dataset_hash():
    games = make_tiny_fixture_with_future_game()
    before = build_features(games, game_id=10, as_of="2024-01-10")
    before_hash = dataset_hash(games)
    edited = edit_score_after_cutoff(games, game_id=1, score=120)
    after = build_features(edited, game_id=10, as_of="2024-01-10")
    assert before != after
    assert dataset_hash(edited) != before_hash


def test_same_day_game_is_excluded_from_cutoff():
    games = make_tiny_fixture_with_future_game()
    same_day = GameRecord(
        game_id=12,
        game_date=date(2024, 1, 10),
        home_team_id=1,
        visitor_team_id=3,
        home_score=130,
        visitor_score=80,
        status="final",
        season=2023,
        source_game_id=12,
    )
    without = build_features(games, game_id=10, as_of="2024-01-10")
    with_same_day = build_features([*games, same_day], game_id=10, as_of="2024-01-10")
    assert without == with_same_day
