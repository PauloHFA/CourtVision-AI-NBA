from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from sqlalchemy.orm import Session

from courtvision.data.models import IngestionRun, utcnow
from courtvision.data.repository import SqlRepository
from courtvision.hashing import canonical_hash
from courtvision.ingestion.normalize import normalize_game, normalize_team
from courtvision.ingestion.quality import QualityReport, build_quality_report


def import_pages(
    session: Session,
    pages: Iterable[tuple[list[dict[str, Any]], int | None]],
    *,
    provider: str = "balldontlie",
    endpoint: str = "GET /v1/games",
    request_params: dict | None = None,
) -> QualityReport:
    """Importa página a página e só avança o cursor depois do commit."""
    repo = SqlRepository(session)
    run = IngestionRun(
        provider=provider,
        endpoint=endpoint,
        started_at=utcnow(),
        state="running",
        request_params_json=request_params or {},
    )
    session.add(run)
    session.commit()

    normalized_all = []
    inserted = 0
    updated = 0
    try:
        for page, next_cursor in pages:
            for raw in page:
                for side in ("home_team", "visitor_team"):
                    if raw.get(side):
                        repo.upsert_team(normalize_team(raw[side]))
                normalized = normalize_game(raw)
                normalized_all.append(normalized)
                outcome = repo.upsert_game(normalized, provider=provider)
                if outcome == "inserted":
                    inserted += 1
                elif outcome == "updated":
                    updated += 1
            run.cursor = None if next_cursor is None else str(next_cursor)
            run.rows_received = len(normalized_all)
            run.payload_hash = canonical_hash(
                [game.source_game_id for game in normalized_all]
            )
            session.commit()
        run.state = "success"
        run.ended_at = utcnow()
        session.commit()
    except Exception as exc:
        session.rollback()
        failed = session.get(IngestionRun, run.id)
        if failed is not None:
            failed.state = "failed"
            failed.error = type(exc).__name__
            failed.ended_at = utcnow()
            session.commit()
        raise

    known = {team.provider_team_id for team in repo.list_teams()}
    return build_quality_report(
        normalized_all,
        known,
        inserted=inserted,
        updated=updated,
    )


def pages_from_games(raw_games: list[dict[str, Any]]) -> Iterator[tuple[list[dict[str, Any]], int | None]]:
    yield raw_games, None
