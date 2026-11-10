from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from courtvision.data.models import Game, Prediction, Team, utcnow
from courtvision.data.records import GameRecord, TeamRecord, dataset_hash, parse_game_date
from courtvision.data.synthetic import build_synthetic_dataset
from courtvision.ingestion.normalize import NormalizedGame, NormalizedTeam


class MemoryRepository:
    mode = "fixtures"

    def __init__(self, teams: list[TeamRecord], games: list[GameRecord]) -> None:
        self._teams = list(teams)
        self._games = list(games)
        self.predictions: list[dict] = []

    @classmethod
    def from_dataset(cls, payload: dict) -> "MemoryRepository":
        teams = [
            TeamRecord(
                id=int(item["id"]),
                provider_team_id=int(item["provider_team_id"]),
                name=item["name"],
                abbreviation=item["abbreviation"],
                city=item.get("city"),
                conference=item.get("conference"),
                division=item.get("division"),
            )
            for item in payload["teams"]
        ]
        games = [
            GameRecord(
                game_id=int(item["id"]),
                game_date=parse_game_date(item["game_date"]),
                home_team_id=int(item["home_team_id"]),
                visitor_team_id=int(item["visitor_team_id"]),
                status=item.get("status", "final"),
                season=int(item.get("season", 0)),
                home_score=item.get("home_score"),
                visitor_score=item.get("visitor_score"),
                source_game_id=item.get("source_game_id"),
                postseason=bool(item.get("postseason", False)),
                provider=item.get("provider", "synthetic"),
            )
            for item in payload["games"]
        ]
        return cls(teams, games)

    @classmethod
    def from_synthetic(cls) -> "MemoryRepository":
        return cls.from_dataset(build_synthetic_dataset())

    def list_teams(self) -> list[TeamRecord]:
        return sorted(self._teams, key=lambda team: team.name)

    def get_team(self, team_id: int) -> TeamRecord | None:
        return next((team for team in self._teams if team.id == team_id), None)

    def list_games(
        self,
        season: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 50,
        offset: int = 0,
        source_game_id: int | None = None,
    ) -> tuple[list[GameRecord], int]:
        rows = self._games
        if season is not None:
            rows = [game for game in rows if game.season == season]
        if date_from is not None:
            rows = [game for game in rows if game.game_date >= date_from]
        if date_to is not None:
            rows = [game for game in rows if game.game_date <= date_to]
        if source_game_id is not None:
            rows = [game for game in rows if game.source_game_id == source_game_id]
        rows = sorted(rows, key=lambda game: (game.game_date, game.game_id))
        return rows[offset : offset + limit], len(rows)

    def all_game_records(self) -> list[GameRecord]:
        return list(self._games)

    def dataset_hash(self) -> str:
        return dataset_hash(self._games)

    def save_prediction(self, payload: dict) -> dict:
        stored = dict(payload)
        stored["id"] = len(self.predictions) + 1
        self.predictions.append(stored)
        return stored


class SqlRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        bind = session.get_bind()
        self.mode = bind.dialect.name if bind is not None else "sql"

    def list_teams(self) -> list[TeamRecord]:
        rows = self.session.scalars(select(Team).order_by(Team.name)).all()
        return [self._team(row) for row in rows]

    def get_team(self, team_id: int) -> TeamRecord | None:
        row = self.session.get(Team, team_id)
        return None if row is None else self._team(row)

    def list_games(
        self,
        season: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 50,
        offset: int = 0,
        source_game_id: int | None = None,
    ) -> tuple[list[GameRecord], int]:
        stmt = select(Game)
        count_stmt = select(func.count()).select_from(Game)
        if season is not None:
            stmt = stmt.where(Game.season == season)
            count_stmt = count_stmt.where(Game.season == season)
        if date_from is not None:
            stmt = stmt.where(Game.game_date >= date_from)
            count_stmt = count_stmt.where(Game.game_date >= date_from)
        if date_to is not None:
            stmt = stmt.where(Game.game_date <= date_to)
            count_stmt = count_stmt.where(Game.game_date <= date_to)
        if source_game_id is not None:
            stmt = stmt.where(Game.source_game_id == source_game_id)
            count_stmt = count_stmt.where(Game.source_game_id == source_game_id)
        total = int(self.session.scalar(count_stmt) or 0)
        rows = self.session.scalars(
            stmt.order_by(Game.game_date, Game.id).offset(offset).limit(limit)
        ).all()
        return [self._game(row) for row in rows], total

    def all_game_records(self) -> list[GameRecord]:
        rows = self.session.scalars(select(Game)).all()
        return [self._game(row) for row in rows]

    def dataset_hash(self) -> str:
        return dataset_hash(self.all_game_records())

    def team_id_for_provider(self, provider_team_id: int) -> int | None:
        row = self.session.scalar(select(Team).where(Team.provider_team_id == provider_team_id))
        return None if row is None else int(row.id)

    def upsert_team(self, team: NormalizedTeam) -> int:
        existing = self.session.scalar(select(Team).where(Team.provider_team_id == team.provider_team_id))
        if existing is None:
            existing = Team(
                provider_team_id=team.provider_team_id,
                name=team.name,
                abbreviation=team.abbreviation,
                city=team.city,
                conference=team.conference,
                division=team.division,
            )
            self.session.add(existing)
            self.session.flush()
            return int(existing.id)
        existing.name = team.name
        existing.abbreviation = team.abbreviation
        existing.city = team.city
        existing.conference = team.conference
        existing.division = team.division
        self.session.flush()
        return int(existing.id)

    def upsert_game(self, game: NormalizedGame, provider: str = "balldontlie") -> str:
        if game.quarantine_reason:
            return "quarantined"
        home_id = self.team_id_for_provider(game.home_provider_team_id)
        away_id = self.team_id_for_provider(game.visitor_provider_team_id)
        if home_id is None or away_id is None:
            return "orphan_team"
        existing = self.session.scalar(
            select(Game).where(Game.provider == provider, Game.source_game_id == game.source_game_id)
        )
        if existing is None:
            self.session.add(
                Game(
                    provider=provider,
                    source_game_id=game.source_game_id,
                    season=game.season,
                    game_date=game.game_date,
                    status=game.status,
                    home_team_id=home_id,
                    visitor_team_id=away_id,
                    home_score=game.home_score,
                    visitor_score=game.visitor_score,
                    postseason=game.postseason,
                )
            )
            self.session.flush()
            return "inserted"
        existing.season = game.season
        existing.game_date = game.game_date
        existing.status = game.status
        existing.home_team_id = home_id
        existing.visitor_team_id = away_id
        existing.home_score = game.home_score
        existing.visitor_score = game.visitor_score
        existing.postseason = game.postseason
        self.session.flush()
        return "updated"

    def save_prediction(self, payload: dict) -> dict:
        row = Prediction(
            target_game_id=payload.get("target_game_id"),
            as_of=payload["as_of"],
            model_version=payload["model_version"],
            p_home=payload.get("p_home_win"),
            p_away=payload.get("p_away_win"),
            feature_hash=payload["feature_hash"],
            feature_version=payload["feature_version"],
            dataset_hash=payload.get("dataset_hash"),
            abstained=payload.get("abstained", False),
            data_quality=payload["data_quality"],
            created_at=utcnow(),
        )
        self.session.add(row)
        self.session.flush()
        stored = dict(payload)
        stored["id"] = int(row.id)
        return stored

    @staticmethod
    def _team(row: Team) -> TeamRecord:
        return TeamRecord(
            id=int(row.id),
            provider_team_id=int(row.provider_team_id),
            name=row.name,
            abbreviation=row.abbreviation,
            city=row.city,
            conference=row.conference,
            division=row.division,
        )

    @staticmethod
    def _game(row: Game) -> GameRecord:
        return GameRecord(
            game_id=int(row.id),
            game_date=row.game_date,
            home_team_id=int(row.home_team_id),
            visitor_team_id=int(row.visitor_team_id),
            status=row.status,
            season=int(row.season),
            home_score=row.home_score,
            visitor_score=row.visitor_score,
            source_game_id=int(row.source_game_id),
            postseason=bool(row.postseason),
            provider=row.provider,
        )
