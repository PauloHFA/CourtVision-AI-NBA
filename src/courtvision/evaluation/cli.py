from __future__ import annotations

import argparse
import json
from pathlib import Path

from courtvision.data.repository import MemoryRepository
from courtvision.evaluation.rag_protocol import evaluate_assistant, protocol_summary
from courtvision.ml.baselines import HomeRateBaseline
from courtvision.ml.predict import PredictionService
from courtvision.rag.assistant import AssistantService


def run_protocol(fixture_path: str, min_prior_games: int = 5) -> dict:
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    repository = MemoryRepository.from_dataset(payload)
    games = repository.all_game_records()
    teams = repository.list_teams()
    predictor = PredictionService(
        games=games,
        known_team_ids={team.id for team in teams},
        model=HomeRateBaseline(),
        min_prior_games=min_prior_games,
        dataset_hash=repository.dataset_hash(),
    )
    assistant = AssistantService(games, teams, predictor)
    return evaluate_assistant(assistant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda o protocolo RAG sintético. Não aceita AC-08.")
    parser.add_argument("--fixture", default="data/fixtures/synthetic_season.json")
    parser.add_argument("--out", default="", help="Se vazio, só imprime o resumo.")
    args = parser.parse_args()
    report = run_protocol(args.fixture)
    summary = protocol_summary(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
