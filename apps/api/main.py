from __future__ import annotations

import logging
from datetime import date

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.limiter import RateLimitMiddleware
from apps.api.schemas import (
    AskRequest,
    AskResponse,
    GameList,
    GameOut,
    PredictionRequest,
    PredictionResponse,
    TeamList,
    TeamOut,
)
from courtvision.config import Settings, get_settings
from courtvision.data.repository import MemoryRepository, SqlRepository
from courtvision.data.session import make_session_factory
from courtvision.errors import CourtVisionError
from courtvision.ml.baselines import HomeRateBaseline
from courtvision.ml.predict import ArtifactModel, PredictionService, load_artifact
from courtvision.evaluation.rag_protocol import evaluate_assistant
from courtvision.rag.assistant import AssistantService

logger = logging.getLogger("courtvision.api")
DISCLAIMER_APP = (
    "CourtVision AI é um estudo educacional. Não é afiliado à NBA, à BALLDONTLIE "
    "ou a qualquer equipe. Não oferece aconselhamento de apostas."
)


def build_repository(settings: Settings):
    if not settings.database_url:
        return MemoryRepository.from_synthetic()
    factory = make_session_factory(settings.database_url)
    session = factory()
    return SqlRepository(session)


def build_services(repository, settings: Settings) -> tuple[PredictionService, AssistantService, str]:
    games = repository.all_game_records()
    teams = repository.list_teams()
    artifact = load_artifact(settings.model_artifact_path)
    if artifact:
        model = ArtifactModel(artifact)
        model_state = "champion" if artifact.get("promoted") else "artifact"
    else:
        model = HomeRateBaseline()
        model_state = "baseline"
    predictor = PredictionService(
        games=games,
        known_team_ids={team.id for team in teams},
        model=model,
        min_prior_games=settings.min_prior_games,
        feature_version=settings.feature_version,
        dataset_hash=repository.dataset_hash(),
    )
    assistant = AssistantService(games, teams, predictor)
    return predictor, assistant, model_state


def probe_ollama(url: str) -> str:
    if not url:
        return "unset"
    try:
        response = httpx.get(url.rstrip("/") + "/api/tags", timeout=0.3)
    except httpx.HTTPError:
        return "unavailable"
    return "ok" if response.status_code == 200 else "unavailable"


def create_app(settings: Settings | None = None, repository=None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="CourtVision AI",
        version="0.1.0",
        description=DISCLAIMER_APP,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.api_rate_limit_per_min)
    if repository is None:
        try:
            repository = build_repository(settings)
        except Exception:
            logger.exception("database_unavailable")
            repository = None
    predictor = assistant = None
    model_state = "unavailable"
    if repository is not None:
        predictor, assistant, model_state = build_services(repository, settings)
    app.state.settings = settings
    app.state.repository = repository
    app.state.predictor = predictor
    app.state.assistant = assistant
    app.state.model_state = model_state

    @app.exception_handler(CourtVisionError)
    async def domain_error(_request: Request, exc: CourtVisionError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = []
        for err in exc.errors():
            loc = ".".join(str(item) for item in err.get("loc", []) if item != "body")
            fields.append({"field": loc, "message": err.get("msg", "inválido")})
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "invalid_schema", "message": "Schema inválido.", "fields": fields}},
        )

    def require_repo():
        if app.state.repository is None or app.state.predictor is None:
            return JSONResponse(
                status_code=503,
                content={"detail": {"code": "source_unavailable", "message": "Fonte de dados indisponível."}},
            )
        return None

    @app.get("/api/v1/health")
    def health() -> JSONResponse:
        database = "down" if app.state.repository is None else app.state.repository.mode
        ollama = probe_ollama(settings.ollama_base_url)
        status = "ok" if app.state.repository is not None else "unavailable"
        code = 200 if status == "ok" else 503
        return JSONResponse(
            status_code=code,
            content={
                "status": status,
                "service": "courtvision-api",
                "components": {
                    "api": "ok",
                    "database": database,
                    "model": app.state.model_state,
                    "ollama": ollama,
                },
                "disclaimer": DISCLAIMER_APP,
            },
        )

    @app.get("/api/v1/teams", response_model=TeamList)
    def teams(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0)):
        blocked = require_repo()
        if blocked:
            return blocked
        rows = app.state.repository.list_teams()
        page = rows[offset : offset + limit]
        return TeamList(
            data=[TeamOut.model_validate(row, from_attributes=True) for row in page],
            meta={"limit": limit, "offset": offset, "count": len(rows)},
        )

    @app.get("/api/v1/games", response_model=GameList)
    def games(
        season: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        source_game_id: int | None = None,
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        blocked = require_repo()
        if blocked:
            return blocked
        rows, total = app.state.repository.list_games(
            season,
            date_from,
            date_to,
            limit,
            offset,
            source_game_id,
        )
        data = [
            GameOut(
                id=row.game_id,
                provider=row.provider,
                source_game_id=row.source_game_id,
                season=row.season,
                game_date=row.game_date.isoformat(),
                status=row.status,
                home_team_id=row.home_team_id,
                visitor_team_id=row.visitor_team_id,
                home_score=row.home_score,
                visitor_score=row.visitor_score,
                postseason=row.postseason,
            )
            for row in rows
        ]
        return GameList(data=data, meta={"limit": limit, "offset": offset, "count": total})

    @app.post("/api/v1/predictions", response_model=PredictionResponse)
    def predict(body: PredictionRequest):
        blocked = require_repo()
        if blocked:
            return blocked
        result = app.state.predictor.predict(
            body.home_team_id,
            body.away_team_id,
            body.as_of,
            game_id=body.game_id,
        )
        stored = app.state.repository.save_prediction(
            {
                **result.as_dict(),
                "as_of": body.as_of,
                "target_game_id": body.game_id,
            }
        )
        payload = result.as_dict()
        if stored.get("id") and payload["prediction_id"] is None:
            payload["prediction_id"] = f"pred_{stored['id']}"
        return payload

    @app.post("/api/v1/assistant/ask", response_model=AskResponse)
    def ask(body: AskRequest):
        blocked = require_repo()
        if blocked:
            return blocked
        if app.state.assistant is None:
            return JSONResponse(
                status_code=503,
                content={"detail": {"code": "assistant_unavailable", "message": "Assistente indisponível."}},
            )
        return app.state.assistant.ask(
            body.question,
            body.as_of,
            body.home_team_id,
            body.away_team_id,
        )

    @app.get("/api/v1/models/current")
    def current_model():
        return {
            "model_version": settings.model_version,
            "feature_version": settings.feature_version,
            "runtime_model": app.state.model_state,
            "min_prior_games": settings.min_prior_games,
            "performance_verified": False,
            "primary_metric": "brier_score",
            "disclaimer": DISCLAIMER_APP,
        }

    @app.get("/api/v1/overview")
    def overview():
        blocked = require_repo()
        if blocked:
            return blocked
        games = app.state.repository.all_game_records()
        seasons: dict[int, dict] = {}
        for game in games:
            bucket = seasons.setdefault(
                int(game.season),
                {"season": int(game.season), "games": 0, "final": 0, "scheduled": 0},
            )
            bucket["games"] += 1
            if game.status == "final":
                bucket["final"] += 1
            else:
                bucket["scheduled"] += 1
        return {
            "teams": len(app.state.repository.list_teams()),
            "games": len(games),
            "seasons": [seasons[key] for key in sorted(seasons)],
            "model_version": settings.model_version,
            "runtime_model": app.state.model_state,
            "feature_version": settings.feature_version,
            "dataset_hash": app.state.repository.dataset_hash(),
            "data_source": app.state.repository.mode,
            "performance_verified": False,
            "disclaimer": DISCLAIMER_APP,
        }

    @app.get("/api/v1/evaluations/latest")
    def latest_evaluation():
        return {
            "available": False,
            "performance_verified": False,
            "message": "Ainda não há avaliação externa verificada. Não trate metas como resultados.",
            "rag_protocol": {
                "path": "/api/v1/evaluations/rag",
                "acceptance_set": False,
                "note": "O protocolo sintético não substitui a avaliação externa.",
            },
        }

    @app.get("/api/v1/evaluations/rag")
    def rag_evaluation():
        blocked = require_repo()
        if blocked:
            return blocked
        if app.state.assistant is None:
            return JSONResponse(
                status_code=503,
                content={"detail": {"code": "assistant_unavailable", "message": "Assistente indisponível."}},
            )
        report = evaluate_assistant(app.state.assistant)
        report["failures"] = report["failures"][:12]
        return report

    return app


app = create_app()
