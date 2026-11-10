"""Gera o fixture sintético versionado. Não chama a API externa."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from courtvision.data.synthetic import build_synthetic_dataset


def main() -> None:
    destination = ROOT / "data" / "fixtures" / "synthetic_season.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_synthetic_dataset()
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"teams={len(payload['teams'])} games={len(payload['games'])} path={destination}")


if __name__ == "__main__":
    main()
