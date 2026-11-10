from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import joblib

from courtvision.data.records import GameRecord
from courtvision.errors import ContestAlreadyFinalError, NotFoundError, SameTeamError
from courtvision.features.builder import build_matchup_features, feature_hash, feature_vector, parse_as_of
from courtvision.features.schema import DISCLAIMER, FEATURE_COLUMNS, FEATURE_VERSION
from courtvision.ml.baselines import HomeRateBaseline, PredictContext


class Predictor(Protocol):
    version: str

    def predict_home(self, features: dict, context: PredictContext) -> float: ...


@dataclass
class PredictionResult:
    p_home_win: float | None
    p_away_win: float | None
    model_version: str
    feature_version: str
    as_of: str
    data_quality: str
    abstained: bool
    factors: list[dict]
    disclaimer: str
    feature_hash: str
    dataset_hash: str | None = None
    prediction_id: str | None = None
    abstain_reason: str | None = None
    home_team_id: int | None = None
    away_team_id: int | None = None

    def as_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "p_home_win": self.p_home_win,
            "p_away_win": self.p_away_win,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "as_of": self.as_of,
            "data_quality": self.data_quality,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "factors": self.factors,
            "disclaimer": self.disclaimer,
            "feature_hash": self.feature_hash,
            "dataset_hash": self.dataset_hash,
        }


FACTOR_LABELS = (
    ("diff_win_rate_last_10", "Diferença de aproveitamento nos últimos 10"),
    ("diff_point_margin_last_10", "Diferença de saldo de pontos nos últimos 10"),
    ("elo_diff", "Diferença de Elo pré-jogo"),
    ("home_days_since_last_game", "Descanso do mandante, em dias"),
    ("away_days_since_last_game", "Descanso do visitante, em dias"),
)


def load_artifact(path: str | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    payload = joblib.load(candidate)
    if not isinstance(payload, dict) or "model" not in payload:
        return None
    return payload


class ArtifactModel:
    def __init__(self, payload: dict) -> None:
        self.model = payload["model"]
        self.columns = list(payload.get("feature_columns") or FEATURE_COLUMNS)
        self.version = str(payload.get("model_version") or "ml-unknown")
        if payload.get("calibrated"):
            method = str(payload.get("calibration_method") or "sigmoid")
            self.version = f"{self.version}+cal-{method}"

    def predict_home(self, features: dict, context: PredictContext) -> float:
        del context
        row = [[float(features[column]) for column in self.columns]]
        probability = float(self.model.predict_proba(row)[0, 1])
        return min(max(probability, 0.0), 1.0)


@dataclass
class PredictionService:
    games: list[GameRecord]
    known_team_ids: set[int]
    model: Predictor | None = None
    min_prior_games: int = 5
    feature_version: str = FEATURE_VERSION
    dataset_hash: str | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = HomeRateBaseline()

    def predict(self, home_id: int, away_id: int, as_of: object, game_id: int | None = None) -> PredictionResult:
        if home_id == away_id:
            raise SameTeamError("Mandante e visitante devem ser equipes diferentes.")
        if home_id not in self.known_team_ids or away_id not in self.known_team_ids:
            raise NotFoundError("Equipe não encontrada.")
        moment = parse_as_of(as_of)  # type: ignore[arg-type]
        self._reject_completed_contest(home_id, away_id, moment.date(), game_id)
        features = build_matchup_features(
            self.games,
            home_team_id=home_id,
            away_team_id=away_id,
            as_of=moment,
            exclude_game_id=game_id,
        )
        digest = feature_hash(features)
        quality = "ok" if float(features["min_prior_games"]) >= self.min_prior_games else "insufficient"
        if quality != "ok":
            return PredictionResult(
                p_home_win=None,
                p_away_win=None,
                model_version=self.model.version,
                feature_version=self.feature_version,
                as_of=moment.isoformat(),
                data_quality=quality,
                abstained=True,
                abstain_reason="Histórico insuficiente para uma estimativa operacional.",
                factors=[],
                disclaimer=DISCLAIMER,
                feature_hash=digest,
                dataset_hash=self.dataset_hash,
                prediction_id=None,
                home_team_id=home_id,
                away_team_id=away_id,
            )
        context = PredictContext(games=self.games, cutoff=date.fromisoformat(str(features["cutoff_date"])))
        probability = float(self.model.predict_home(features, context))
        probability = min(max(probability, 0.0), 1.0)
        away = 1.0 - probability
        return PredictionResult(
            p_home_win=round(probability, 6),
            p_away_win=round(away, 6),
            model_version=self.model.version,
            feature_version=self.feature_version,
            as_of=moment.isoformat(),
            data_quality=quality,
            abstained=False,
            factors=_factors(features),
            disclaimer=DISCLAIMER,
            feature_hash=digest,
            dataset_hash=self.dataset_hash,
            prediction_id=f"pred_{uuid4().hex[:12]}",
            home_team_id=home_id,
            away_team_id=away_id,
        )

    def _reject_completed_contest(
        self,
        home_id: int,
        away_id: int,
        on_date: date,
        game_id: int | None,
    ) -> None:
        for game in self.games:
            if game.status != "final":
                continue
            same_pair = game.home_team_id == home_id and game.visitor_team_id == away_id
            targeted = game_id is not None and game.game_id == game_id
            if targeted or (same_pair and game.game_date == on_date):
                raise ContestAlreadyFinalError(
                    "Esta partida já foi concluída. O endpoint de previsão futura não reescreve o resultado."
                )


def _factors(features: dict) -> list[dict]:
    rows = []
    for key, label in FACTOR_LABELS:
        rows.append({"feature": key, "label": label, "value": features.get(key)})
    return rows


def vector_ready(features: dict) -> list[float]:
    return feature_vector(features)
