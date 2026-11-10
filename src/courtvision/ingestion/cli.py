from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sqlalchemy.orm import Session

from courtvision.config import get_settings
from courtvision.data.models import Base
from courtvision.data.session import make_engine, make_session_factory
from courtvision.ingestion.client import BallDontLieClient
from courtvision.ingestion.normalize import normalize_game, normalize_team
from courtvision.ingestion.quality import build_quality_report
from courtvision.ingestion.service import import_pages, pages_from_games


def fixture_to_raw_games(payload: dict) -> list[dict]:
    teams_by_id = {int(t["id"]): t for t in payload["teams"]}
    raw_games = []
    for item in payload["games"]:
        ht = teams_by_id[int(item["home_team_id"])]
        vt = teams_by_id[int(item["visitor_team_id"])]
        raw_games.append(
            {
                "id": item["source_game_id"],
                "date": item["game_date"],
                "season": item["season"],
                "status": item["status"],
                "postseason": item.get("postseason", False),
                "home_team_score": item.get("home_score"),
                "visitor_team_score": item.get("visitor_score"),
                "home_team": {
                    "id": ht["provider_team_id"],
                    "full_name": ht["name"],
                    "name": ht["name"],
                    "abbreviation": ht["abbreviation"],
                    "city": ht.get("city"),
                    "conference": ht.get("conference"),
                    "division": ht.get("division"),
                },
                "visitor_team": {
                    "id": vt["provider_team_id"],
                    "full_name": vt["name"],
                    "name": vt["name"],
                    "abbreviation": vt["abbreviation"],
                    "city": vt.get("city"),
                    "conference": vt.get("conference"),
                    "division": vt.get("division"),
                },
            }
        )
    return raw_games


def import_fixture(session: Session, path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_games = fixture_to_raw_games(payload)
    provider = payload["games"][0].get("provider", "synthetic") if payload["games"] else "synthetic"
    report = import_pages(
        session,
        pages_from_games(raw_games),
        provider=provider,
        endpoint=f"fixture:{path}",
        request_params={"fixture": path},
    )
    return report.as_dict()


def quality_from_fixture(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    teams = [normalize_team({"id": team["provider_team_id"], **team}) for team in payload["teams"]]
    known = {team.provider_team_id for team in teams}
    games = []
    for item in payload["games"]:
        games.append(
            normalize_game(
                {
                    "id": item["source_game_id"],
                    "date": item["game_date"],
                    "season": item["season"],
                    "status": item["status"],
                    "postseason": item.get("postseason", False),
                    "home_team_score": item.get("home_score"),
                    "visitor_team_score": item.get("visitor_score"),
                    "home_team": {"id": _provider_for(payload, item["home_team_id"])},
                    "visitor_team": {"id": _provider_for(payload, item["visitor_team_id"])},
                }
            )
        )
    return build_quality_report(games, known).as_dict()


def _provider_for(payload: dict, team_id: int) -> int:
    for team in payload["teams"]:
        if int(team["id"]) == int(team_id):
            return int(team["provider_team_id"])
    raise KeyError(team_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa equipes e jogos autorizados.")
    parser.add_argument("--season", type=int)
    parser.add_argument("--fixture")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--out", default="reports/quality_latest.json")
    args = parser.parse_args()
    settings = get_settings()
    database_url = args.database_url or settings.database_url or os.environ.get("DATABASE_URL", "")
    if args.fixture:
        if database_url:
            engine = make_engine(database_url)
            Base.metadata.create_all(engine)
            factory = make_session_factory(database_url)
            with factory() as session:
                report = import_fixture(session, args.fixture)
        else:
            report = quality_from_fixture(args.fixture)
    elif args.season is not None:
        if not database_url:
            raise SystemExit("DATABASE_URL é obrigatório para importar da API.")
        interval = 60.0 / max(settings.ingestion_rate_per_min, 1)
        engine = make_engine(database_url)
        Base.metadata.create_all(engine)
        factory = make_session_factory(database_url)
        with BallDontLieClient(settings.bdl_api_key, min_interval_s=interval) as client:
            with factory() as session:
                report_obj = import_pages(
                    session,
                    client.iter_game_pages(args.season),
                    request_params={"season": args.season},
                )
                report = report_obj.as_dict()
    else:
        raise SystemExit("Informe --fixture ou --season.")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "games": report["games"], "out": str(destination)}))


if __name__ == "__main__":
    main()
