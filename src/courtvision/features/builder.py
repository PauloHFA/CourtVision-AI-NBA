from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from courtvision.data.records import GameRecord, event_key
from courtvision.errors import InvalidAsOfError, NotFoundError
from courtvision.features.schema import (
    ELO_HOME_ADVANTAGE,
    ELO_K,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    FINAL_STATUS,
    INITIAL_ELO,
    NEUTRAL_POINTS,
    NEUTRAL_REST_DAYS,
    NEUTRAL_WIN_RATE,
)
from courtvision.hashing import canonical_hash


@dataclass(frozen=True)
class TeamGameView:
    game_date: date
    season: int
    points_for: int
    points_against: int
    won: bool
    was_home: bool


def parse_as_of(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            raise InvalidAsOfError("as_of é obrigatório.")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidAsOfError("as_of não é uma data ISO válida.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _round(value: float) -> float:
    return round(float(value), 6)


def _usable(game: GameRecord, cutoff: date) -> bool:
    return (
        game.status == FINAL_STATUS
        and not game.postseason
        and game.game_date < cutoff
        and game.home_score is not None
        and game.visitor_score is not None
        and game.home_score >= 0
        and game.visitor_score >= 0
    )


def _view(game: GameRecord, team_id: int) -> TeamGameView:
    if game.home_team_id == team_id:
        points_for = int(game.home_score or 0)
        points_against = int(game.visitor_score or 0)
        was_home = True
    else:
        points_for = int(game.visitor_score or 0)
        points_against = int(game.home_score or 0)
        was_home = False
    return TeamGameView(
        game_date=game.game_date,
        season=game.season,
        points_for=points_for,
        points_against=points_against,
        won=points_for > points_against,
        was_home=was_home,
    )


def elo_before(games: list[GameRecord], cutoff: date) -> dict[int, float]:
    ratings: dict[int, float] = {}
    ordered = sorted((game for game in games if _usable(game, cutoff)), key=event_key)
    for game in ordered:
        home = ratings.get(game.home_team_id, INITIAL_ELO)
        away = ratings.get(game.visitor_team_id, INITIAL_ELO)
        expected = 1.0 / (1.0 + 10 ** ((away - (home + ELO_HOME_ADVANTAGE)) / 400.0))
        if game.home_score == game.visitor_score:
            score = 0.5
        elif int(game.home_score or 0) > int(game.visitor_score or 0):
            score = 1.0
        else:
            score = 0.0
        ratings[game.home_team_id] = home + ELO_K * (score - expected)
        ratings[game.visitor_team_id] = away + ELO_K * ((1.0 - score) - (1.0 - expected))
    return ratings


def _window(rows: list[TeamGameView], size: int) -> dict[str, float | int | bool]:
    sample = rows[:size]
    if not sample:
        return {
            "win_rate": NEUTRAL_WIN_RATE,
            "points_for": NEUTRAL_POINTS,
            "points_against": NEUTRAL_POINTS,
            "margin": 0.0,
            "n": 0,
            "imputed": True,
        }
    count = len(sample)
    wins = sum(1 for row in sample if row.won)
    points_for = sum(row.points_for for row in sample) / count
    points_against = sum(row.points_against for row in sample) / count
    return {
        "win_rate": wins / count,
        "points_for": points_for,
        "points_against": points_against,
        "margin": points_for - points_against,
        "n": count,
        "imputed": False,
    }


def _rate(rows: list[TeamGameView]) -> float:
    if not rows:
        return NEUTRAL_WIN_RATE
    return sum(1 for row in rows if row.won) / len(rows)


def _win_streak(rows: list[TeamGameView]) -> int:
    streak = 0
    for row in rows:
        if not row.won:
            break
        streak += 1
    return streak


def _team_block(rows: list[TeamGameView], cutoff: date, elo: float, season: int) -> dict[str, float]:
    newest_first = list(reversed(rows))
    last_5 = _window(newest_first, 5)
    last_10 = _window(newest_first, 10)
    imputed = not rows
    if rows:
        rest = float((cutoff - newest_first[0].game_date).days)
    else:
        rest = NEUTRAL_REST_DAYS
    season_games = sum(1 for row in rows if row.season == season)
    return {
        "win_rate_last_5": float(last_5["win_rate"]),
        "win_rate_last_10": float(last_10["win_rate"]),
        "current_win_streak": float(_win_streak(newest_first)),
        "points_for_avg_5": float(last_5["points_for"]),
        "points_for_avg_10": float(last_10["points_for"]),
        "points_against_avg_5": float(last_5["points_against"]),
        "points_against_avg_10": float(last_10["points_against"]),
        "margin_last_10": float(last_10["margin"]),
        "elo_pre": elo,
        "days_since_last_game": rest,
        "back_to_back": 1.0 if rest == 1.0 else 0.0,
        "win_rate": _rate(rows),
        "game_number_in_season": float(season_games + 1),
        "imputed": 1.0 if imputed else 0.0,
        "short_history": 1.0 if len(rows) < 10 else 0.0,
        "prior_games": float(len(rows)),
        "home_role_rate": _rate([row for row in rows if row.was_home]),
        "road_role_rate": _rate([row for row in rows if not row.was_home]),
    }


def infer_season(as_of: datetime, explicit: int | None = None) -> int:
    if explicit is not None:
        return explicit
    # Temporada rotulada pelo ano de início, como na fonte de jogos.
    return as_of.year if as_of.month >= 8 else as_of.year - 1


def build_matchup_features(
    games: list[GameRecord],
    home_team_id: int,
    away_team_id: int,
    as_of: datetime | date | str,
    season: int | None = None,
    exclude_game_id: int | None = None,
) -> dict[str, float | str | int]:
    moment = parse_as_of(as_of)
    cutoff = moment.date()
    if exclude_game_id is not None:
        target = next((game for game in games if game.game_id == exclude_game_id), None)
        if target is not None:
            cutoff = min(cutoff, target.game_date)
    history = [
        game
        for game in games
        if game.game_id != exclude_game_id and _usable(game, cutoff)
    ]
    ratings = elo_before(history, cutoff)
    # elo_before já filtra por cutoff; passar history não muda o resultado.
    home_rows = [_view(game, home_team_id) for game in sorted(history, key=event_key) if home_team_id in (game.home_team_id, game.visitor_team_id)]
    away_rows = [_view(game, away_team_id) for game in sorted(history, key=event_key) if away_team_id in (game.home_team_id, game.visitor_team_id)]
    season_id = infer_season(moment, season)
    home = _team_block(home_rows, cutoff, ratings.get(home_team_id, INITIAL_ELO), season_id)
    away = _team_block(away_rows, cutoff, ratings.get(away_team_id, INITIAL_ELO), season_id)
    features: dict[str, float | str | int] = {
        "home_win_rate_at_home": _round(float(home["home_role_rate"])),
        "away_win_rate_on_road": _round(float(away["road_role_rate"])),
        "home_win_rate_last_5": _round(float(home["win_rate_last_5"])),
        "away_win_rate_last_5": _round(float(away["win_rate_last_5"])),
        "home_win_rate_last_10": _round(float(home["win_rate_last_10"])),
        "away_win_rate_last_10": _round(float(away["win_rate_last_10"])),
        "home_current_win_streak": _round(float(home["current_win_streak"])),
        "away_current_win_streak": _round(float(away["current_win_streak"])),
        "home_points_for_avg_5": _round(float(home["points_for_avg_5"])),
        "away_points_for_avg_5": _round(float(away["points_for_avg_5"])),
        "home_points_for_avg_10": _round(float(home["points_for_avg_10"])),
        "away_points_for_avg_10": _round(float(away["points_for_avg_10"])),
        "home_points_against_avg_5": _round(float(home["points_against_avg_5"])),
        "away_points_against_avg_5": _round(float(away["points_against_avg_5"])),
        "home_points_against_avg_10": _round(float(home["points_against_avg_10"])),
        "away_points_against_avg_10": _round(float(away["points_against_avg_10"])),
        "diff_win_rate_last_10": _round(float(home["win_rate_last_10"]) - float(away["win_rate_last_10"])),
        "diff_point_margin_last_10": _round(float(home["margin_last_10"]) - float(away["margin_last_10"])),
        "home_elo_pre": _round(float(home["elo_pre"])),
        "away_elo_pre": _round(float(away["elo_pre"])),
        "elo_diff": _round(float(home["elo_pre"]) - float(away["elo_pre"])),
        "home_days_since_last_game": _round(float(home["days_since_last_game"])),
        "away_days_since_last_game": _round(float(away["days_since_last_game"])),
        "home_back_to_back": float(home["back_to_back"]),
        "away_back_to_back": float(away["back_to_back"]),
        "home_win_rate": _round(float(home["win_rate"])),
        "away_win_rate": _round(float(away["win_rate"])),
        "season_id": float(season_id),
        "home_game_number_in_season": float(home["game_number_in_season"]),
        "away_game_number_in_season": float(away["game_number_in_season"]),
        "home_imputed": float(home["imputed"]),
        "away_imputed": float(away["imputed"]),
        "home_short_history": float(home["short_history"]),
        "away_short_history": float(away["short_history"]),
        "min_prior_games": float(min(home["prior_games"], away["prior_games"])),
        "cutoff_date": cutoff.isoformat(),
        "feature_version": FEATURE_VERSION,
        "home_prior_games": int(home["prior_games"]),
        "away_prior_games": int(away["prior_games"]),
    }
    return features


def build_features(
    games: list[GameRecord],
    game_id: int,
    as_of: datetime | date | str,
) -> dict[str, float | str | int]:
    target = next((game for game in games if game.game_id == game_id), None)
    if target is None:
        raise NotFoundError(f"Jogo {game_id} não encontrado.")
    return build_matchup_features(
        games,
        home_team_id=target.home_team_id,
        away_team_id=target.visitor_team_id,
        as_of=as_of,
        season=target.season,
        exclude_game_id=target.game_id,
    )


def feature_vector(features: dict) -> list[float]:
    return [float(features[column]) for column in FEATURE_COLUMNS]


def feature_hash(features: dict) -> str:
    model_payload = {column: features[column] for column in FEATURE_COLUMNS}
    model_payload["cutoff_date"] = features.get("cutoff_date")
    return canonical_hash(model_payload)
