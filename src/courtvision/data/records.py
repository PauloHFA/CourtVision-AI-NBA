from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from courtvision.hashing import canonical_hash


@dataclass(frozen=True)
class TeamRecord:
    id: int
    provider_team_id: int
    name: str
    abbreviation: str
    city: str | None = None
    conference: str | None = None
    division: str | None = None


@dataclass(frozen=True)
class GameRecord:
    game_id: int
    game_date: date
    home_team_id: int
    visitor_team_id: int
    status: str = "final"
    season: int = 0
    home_score: int | None = None
    visitor_score: int | None = None
    source_game_id: int | None = None
    postseason: bool = False
    provider: str = "synthetic"

    def __post_init__(self) -> None:
        if self.home_team_id == self.visitor_team_id:
            raise ValueError("home_team_id must differ from visitor_team_id")


def event_key(game: GameRecord) -> tuple:
    source = game.source_game_id if game.source_game_id is not None else game.game_id
    return (game.game_date, source, game.game_id)


def dataset_hash(games: list[GameRecord]) -> str:
    payload = [
        {
            "game_id": game.game_id,
            "source_game_id": game.source_game_id,
            "provider": game.provider,
            "game_date": game.game_date.isoformat(),
            "status": game.status,
            "season": game.season,
            "home_team_id": game.home_team_id,
            "visitor_team_id": game.visitor_team_id,
            "home_score": game.home_score,
            "visitor_score": game.visitor_score,
            "postseason": game.postseason,
        }
        for game in sorted(games, key=lambda item: item.game_id)
    ]
    return canonical_hash(payload)


def parse_game_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
