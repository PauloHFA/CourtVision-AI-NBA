from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from courtvision.data.records import GameRecord
from courtvision.features.builder import _usable


@dataclass
class PredictContext:
    games: list[GameRecord]
    cutoff: date


class ConstantProbability:
    version = "baseline-constant-0.1.0"

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability
        self.version = f"baseline-constant-{probability:.2f}"

    def predict_home(self, features: dict, context: PredictContext) -> float:
        del features, context
        return self.probability


class HomeRateBaseline:
    """Taxa histórica de vitória do mandante, só com jogos anteriores ao corte."""

    version = "baseline-home-rate-0.1.0"

    def predict_home(self, features: dict, context: PredictContext) -> float:
        del features
        finals = [game for game in context.games if _usable(game, context.cutoff)]
        if not finals:
            return 0.5
        wins = sum(1 for game in finals if int(game.home_score or 0) > int(game.visitor_score or 0))
        return wins / len(finals)
