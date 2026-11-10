from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[int]:
    return mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = _pk()
    provider_team_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(8), nullable=False)
    city: Mapped[str | None] = mapped_column(Text, nullable=True)
    conference: Mapped[str | None] = mapped_column(Text, nullable=True)
    division: Mapped[str | None] = mapped_column(Text, nullable=True)


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        UniqueConstraint("provider", "source_game_id", name="uq_games_provider_source"),
        CheckConstraint("home_team_id <> visitor_team_id", name="ck_games_distinct_teams"),
        CheckConstraint("home_score IS NULL OR home_score >= 0", name="ck_games_home_score"),
        CheckConstraint("visitor_score IS NULL OR visitor_score >= 0", name="ck_games_visitor_score"),
        Index("idx_games_date", "game_date"),
        Index("idx_games_teams", "home_team_id", "visitor_team_id"),
    )

    id: Mapped[int] = _pk()
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="balldontlie")
    source_game_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    visitor_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    visitor_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    postseason: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = _pk()
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cursor: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    rows_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_params_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"

    id: Mapped[int] = _pk()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    filter_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    terms_checked_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="balldontlie")


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id: Mapped[int] = _pk()
    game_id: Mapped[int | None] = mapped_column(ForeignKey("games.id"), nullable=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    visitor_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    features_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = _pk()
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_snapshots.id"), nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = _pk()
    target_game_id: Mapped[int | None] = mapped_column(ForeignKey("games.id"), nullable=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    p_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data_quality: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = _pk()
    doc_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)
    event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = _pk()
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
