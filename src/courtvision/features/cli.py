from __future__ import annotations

import argparse
import json
from pathlib import Path

from courtvision.data.repository import MemoryRepository
from courtvision.features.builder import build_features, parse_as_of
from courtvision.hashing import canonical_hash


def build_preview(fixture_path: str, as_of: str, limit: int = 20) -> dict:
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    repo = MemoryRepository.from_dataset(payload)
    moment = parse_as_of(as_of)
    rows = []
    for game in repo.all_game_records():
        if game.game_date >= moment.date() or game.status != "final":
            continue
        features = build_features(repo.all_game_records(), game.game_id, moment)
        rows.append(
            {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "min_prior_games": features["min_prior_games"],
                "feature_hash": canonical_hash(
                    {key: features[key] for key in ("cutoff_date", "diff_win_rate_last_10", "elo_diff")}
                ),
            }
        )
        if len(rows) >= limit:
            break
    return {"as_of": moment.isoformat(), "count": len(rows), "preview": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera uma prévia de features pré-jogo.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--fixture", default="data/fixtures/synthetic_season.json")
    parser.add_argument("--out", default="reports/features_preview.json")
    args = parser.parse_args()
    report = build_preview(args.fixture, args.as_of)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"features={report['count']} out={destination}")


if __name__ == "__main__":
    main()
