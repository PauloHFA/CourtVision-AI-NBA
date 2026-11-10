import httpx
import pytest

from courtvision.errors import PermanentAuthError
from courtvision.ingestion.client import BallDontLieClient
from courtvision.ingestion.normalize import normalize_game, normalize_status


def test_status_final_is_normalized():
    assert normalize_status("Final") == "final"
    assert normalize_status("Final/OT") == "final"


def test_same_team_is_quarantined():
    game = normalize_game(
        {
            "id": 9,
            "date": "2024-01-02",
            "season": 2023,
            "status": "Final",
            "home_team_score": 10,
            "visitor_team_score": 9,
            "home_team": {"id": 3, "full_name": "A", "abbreviation": "AAA"},
            "visitor_team": {"id": 3, "full_name": "A", "abbreviation": "AAA"},
        }
    )
    assert game.quarantine_reason == "same_team"


def test_auth_failure_is_not_retried():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "unauthorized"})

    sleeps: list[float] = []
    client = BallDontLieClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
        sleeper=sleeps.append,
    )
    with pytest.raises(PermanentAuthError):
        client._get("/games", {"seasons[]": 2023})
    assert calls["n"] == 1
    assert sleeps == []
    client.close()


def test_pagination_and_retry_after():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        if calls["n"] == 2:
            return httpx.Response(200, json={"data": [{"id": 1}], "meta": {"next_cursor": 20}})
        return httpx.Response(200, json={"data": [{"id": 2}], "meta": {"next_cursor": None}})

    client = BallDontLieClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        min_interval_s=0,
        sleeper=lambda _delay: None,
    )
    rows = list(client.iter_games(2023))
    assert [row["id"] for row in rows] == [1, 2]
    assert calls["n"] == 3
    client.close()


def test_import_fixture_to_db(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from courtvision.data.models import Base, Game, Team
    from courtvision.ingestion.cli import import_fixture

    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        report = import_fixture(session, "data/fixtures/synthetic_season.json")
        assert report["ok"] is True
        assert report["inserted"] == 73
        assert session.query(Game).count() == 73
        assert session.query(Team).count() == 4

        # Second run is idempotent
        second_report = import_fixture(session, "data/fixtures/synthetic_season.json")
        assert second_report["ok"] is True
        assert second_report["inserted"] == 0
        assert second_report["updated"] == 73

