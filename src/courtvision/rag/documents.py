from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from courtvision.data.records import GameRecord, TeamRecord
from courtvision.hashing import canonical_hash
from courtvision.rag.prompts import HOSTILE_MARKERS


@dataclass(frozen=True)
class EvidenceDocument:
    doc_id: str
    doc_type: str
    source_ref: str
    text: str
    available_at: datetime
    event_at: datetime | None = None
    team_ids: tuple[int, ...] = ()
    hostile: bool = False
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            object.__setattr__(self, "checksum", canonical_hash(self.text))


def is_hostile(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in HOSTILE_MARKERS)


def _at(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def game_documents(games: list[GameRecord], teams: list[TeamRecord]) -> list[EvidenceDocument]:
    names = {team.id: team.name for team in teams}
    documents = []
    for game in games:
        home = names.get(game.home_team_id, str(game.home_team_id))
        away = names.get(game.visitor_team_id, str(game.visitor_team_id))
        source = game.source_game_id or game.game_id
        if game.status == "final" and game.home_score is not None and game.visitor_score is not None:
            text = (
                f"{home} enfrentou {away} em {game.game_date.isoformat()}: "
                f"{game.home_score} a {game.visitor_score}."
            )
        else:
            text = f"{home} x {away} em {game.game_date.isoformat()} ainda não possui resultado final."
        documents.append(
            EvidenceDocument(
                doc_id=f"game:{source}",
                doc_type="game_summary",
                source_ref=f"game:{source}",
                text=text,
                event_at=_at(game.game_date),
                available_at=_at(game.game_date + timedelta(days=1)),
                team_ids=(game.home_team_id, game.visitor_team_id),
                hostile=is_hostile(text),
            )
        )
    documents.append(
        EvidenceDocument(
            doc_id="method:features-1",
            doc_type="methodology",
            source_ref="method:features-1",
            text=(
                "As features usam apenas jogos finalizados com data anterior ao corte as_of. "
                "O modelo produz a probabilidade; o texto não altera esse número."
            ),
            event_at=_at(date(2026, 11, 8)),
            available_at=_at(date(2020, 1, 1)),
            hostile=False,
        )
    )
    return documents


def trend_fact(team_name: str, wins: int, sample: int, as_of: date, source_ids: list[str]) -> dict:
    return {
        "metric": "wins_in_sample",
        "value": wins,
        "sample": sample,
        "team": team_name,
        "as_of": as_of.isoformat(),
        "source_ids": source_ids,
        "text": f"Nos {sample} jogos anteriores a {as_of.isoformat()}, {team_name} teve {wins} vitórias.",
    }
