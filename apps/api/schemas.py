from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    provider_team_id: int
    name: str
    abbreviation: str
    city: str | None = None
    conference: str | None = None
    division: str | None = None


class GameOut(BaseModel):
    id: int
    provider: str
    source_game_id: int | None
    season: int
    game_date: str
    status: str
    home_team_id: int
    visitor_team_id: int
    home_score: int | None
    visitor_score: int | None
    postseason: bool


class PageMeta(BaseModel):
    limit: int
    offset: int
    count: int


class TeamList(BaseModel):
    data: list[TeamOut]
    meta: PageMeta


class GameList(BaseModel):
    data: list[GameOut]
    meta: PageMeta


class PredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    as_of: datetime
    game_id: int | None = None


class FactorOut(BaseModel):
    feature: str
    label: str
    value: float | None = None


class PredictionResponse(BaseModel):
    prediction_id: str | None
    p_home_win: float | None
    p_away_win: float | None
    model_version: str
    feature_version: str
    as_of: str
    data_quality: str
    abstained: bool
    abstain_reason: str | None = None
    factors: list[FactorOut]
    disclaimer: str
    feature_hash: str
    dataset_hash: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    as_of: datetime
    home_team_id: int | None = None
    away_team_id: int | None = None


class AskResponse(BaseModel):
    answer: str
    facts: list[dict]
    citations: list[dict]
    model_version: str | None
    p_home_win: float | None = None
    dataset_version: str | None
    insufficient_evidence: bool
    llm_used: bool
