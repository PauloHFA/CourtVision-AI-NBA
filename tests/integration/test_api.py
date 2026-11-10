from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.api.main import create_app
from courtvision.config import Settings
from courtvision.data.models import Base, Game
from courtvision.data.repository import MemoryRepository, SqlRepository
from courtvision.ingestion.service import import_pages, pages_from_games


def _settings() -> Settings:
    return Settings(
        database_url="",
        ollama_base_url="",
        bdl_api_key="",
        api_rate_limit_per_min=1000,
        min_prior_games=5,
        model_artifact_path="models/does-not-exist.joblib",
    )


def _client() -> TestClient:
    app = create_app(settings=_settings(), repository=MemoryRepository.from_synthetic())
    return TestClient(app)


def test_health_lists_components():
    response = _client().get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["components"]["api"] == "ok"
    assert body["components"]["database"] == "fixtures"
    assert body["components"]["model"] == "baseline"
    assert "não é afiliado" in body["disclaimer"].lower() or "Nao" in body["disclaimer"] or "Não" in body["disclaimer"]


def test_teams_and_prediction_contract():
    client = _client()
    teams = client.get("/api/v1/teams")
    assert teams.status_code == 200
    ids = [row["id"] for row in teams.json()["data"]]
    assert len(ids) >= 2
    same = client.post(
        "/api/v1/predictions",
        json={"home_team_id": ids[0], "away_team_id": ids[0], "as_of": "2024-04-01T18:00:00Z"},
    )
    assert same.status_code == 400
    missing = client.post(
        "/api/v1/predictions",
        json={"home_team_id": 999, "away_team_id": ids[0], "as_of": "2024-04-01T18:00:00Z"},
    )
    assert missing.status_code == 404
    predicted = client.post(
        "/api/v1/predictions",
        json={"home_team_id": ids[0], "away_team_id": ids[1], "as_of": "2024-04-01T18:00:00Z"},
    )
    assert predicted.status_code == 200
    body = predicted.json()
    assert body["abstained"] is False
    assert abs(body["p_home_win"] + body["p_away_win"] - 1) < 1e-6
    assert body["model_version"]
    assert body["disclaimer"]
    early = client.post(
        "/api/v1/predictions",
        json={"home_team_id": ids[0], "away_team_id": ids[1], "as_of": "2020-01-01T00:00:00Z"},
    )
    assert early.status_code == 200
    assert early.json()["abstained"] is True
    assert early.json()["p_home_win"] is None


def test_invalid_schema_is_422():
    response = _client().post("/api/v1/predictions", json={"home_team_id": 1})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_schema"


def test_assistant_returns_citations_without_inventing_probability():
    client = _client()
    response = client.post(
        "/api/v1/assistant/ask",
        json={
            "question": "Quais jogos anteriores sustentam a análise?",
            "as_of": "2024-04-01T18:00:00Z",
            "home_team_id": 1,
            "away_team_id": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["llm_used"] is False
    assert body["citations"]
    assert body["p_home_win"] is None or 0 <= body["p_home_win"] <= 1


def test_reingest_does_not_duplicate_source_ids():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    raw = {
        "id": 77,
        "date": "2023-11-01",
        "season": 2023,
        "status": "Final",
        "postseason": False,
        "home_team_score": 101,
        "visitor_team_score": 99,
        "home_team": {"id": 11, "full_name": "Alpha", "abbreviation": "ALP"},
        "visitor_team": {"id": 12, "full_name": "Beta", "abbreviation": "BET"},
    }
    with Session() as session:
        first = import_pages(session, pages_from_games([raw]))
        second = import_pages(session, pages_from_games([raw]))
        count = session.scalar(select(Game).where(Game.source_game_id == 77))
        assert first.inserted == 1
        assert second.updated == 1
        assert count is not None
        assert session.query(Game).count() == 1
        repo = SqlRepository(session)
        assert repo.get_team(repo.list_teams()[0].id) is not None


def test_models_endpoint_does_not_claim_verified_performance():
    response = _client().get("/api/v1/models/current")
    assert response.status_code == 200
    assert response.json()["performance_verified"] is False
    latest = _client().get("/api/v1/evaluations/latest")
    assert latest.json()["available"] is False
    assert latest.json()["performance_verified"] is False


def test_overview_and_game_lookup_support_the_dashboard():
    client = _client()
    overview = client.get("/api/v1/overview")
    assert overview.status_code == 200
    body = overview.json()
    assert body["performance_verified"] is False
    assert body["teams"] >= 2
    assert body["games"] > 0
    assert body["seasons"]
    found = client.get("/api/v1/games", params={"source_game_id": 5001})
    assert found.status_code == 200
    rows = found.json()["data"]
    assert len(rows) == 1
    assert rows[0]["source_game_id"] == 5001
    missing = client.get("/api/v1/games", params={"source_game_id": 999999})
    assert missing.json()["meta"]["count"] == 0


def test_openapi_lists_the_mvp_routes():
    schema = _client().get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    for path in (
        "/api/v1/health",
        "/api/v1/teams",
        "/api/v1/games",
        "/api/v1/predictions",
        "/api/v1/assistant/ask",
        "/api/v1/models/current",
        "/api/v1/overview",
        "/api/v1/evaluations/latest",
        "/api/v1/evaluations/rag",
    ):
        assert path in paths
    names = {item["name"] for item in paths["/api/v1/games"]["get"]["parameters"]}
    assert "source_game_id" in names
