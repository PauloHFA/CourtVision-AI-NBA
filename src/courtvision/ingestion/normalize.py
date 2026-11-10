from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class NormalizedTeam:
    provider_team_id: int
    name: str
    abbreviation: str
    city: str | None = None
    conference: str | None = None
    division: str | None = None


@dataclass(frozen=True)
class NormalizedGame:
    source_game_id: int
    season: int
    game_date: date
    status: str
    home_provider_team_id: int
    visitor_provider_team_id: int
    home_score: int | None
    visitor_score: int | None
    postseason: bool
    quarantine_reason: str | None = None


def normalize_status(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if "final" in value:
        return "final"
    return value or "unknown"


def _score(payload: dict, *keys: str) -> int | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return int(payload[key])
    return None


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw)[:10])


def normalize_team(payload: dict) -> NormalizedTeam:
    name = payload.get("full_name") or payload.get("name") or ""
    abbreviation = payload.get("abbreviation") or ""
    return NormalizedTeam(
        provider_team_id=int(payload["id"]),
        name=str(name),
        abbreviation=str(abbreviation)[:8],
        city=payload.get("city"),
        conference=payload.get("conference"),
        division=payload.get("division"),
    )


def normalize_game(payload: dict) -> NormalizedGame:
    home = payload.get("home_team") or {}
    visitor = payload.get("visitor_team") or {}
    status = normalize_status(payload.get("status"))
    home_score = _score(payload, "home_team_score", "home_score")
    visitor_score = _score(payload, "visitor_team_score", "visitor_score")
    home_id = int(home["id"])
    visitor_id = int(visitor["id"])
    reason = None
    if home_id == visitor_id:
        reason = "same_team"
    elif status == "final" and (home_score is None or visitor_score is None):
        reason = "final_missing_score"
    elif home_score is not None and home_score < 0:
        reason = "negative_score"
    elif visitor_score is not None and visitor_score < 0:
        reason = "negative_score"
    try:
        game_date = _parse_date(payload.get("date") or payload["game_date"])
    except (KeyError, ValueError):
        game_date = date(1970, 1, 1)
        reason = reason or "invalid_date"
    return NormalizedGame(
        source_game_id=int(payload["id"] if "id" in payload else payload["source_game_id"]),
        season=int(payload.get("season") or game_date.year),
        game_date=game_date,
        status=status,
        home_provider_team_id=home_id,
        visitor_provider_team_id=visitor_id,
        home_score=home_score,
        visitor_score=visitor_score,
        postseason=bool(payload.get("postseason", False)),
        quarantine_reason=reason,
    )
