from __future__ import annotations

from datetime import date, timedelta


def build_synthetic_dataset() -> dict:
    """Confrontos fictícios para testes e demo. Não são dados da NBA."""
    teams = [
        {
            "id": 1,
            "provider_team_id": 101,
            "name": "Harbor Hawks",
            "abbreviation": "HRH",
            "city": "Harbor",
            "conference": "Leste",
            "division": "Atlântica",
        },
        {
            "id": 2,
            "provider_team_id": 102,
            "name": "Cedar Wolves",
            "abbreviation": "CDW",
            "city": "Cedar",
            "conference": "Leste",
            "division": "Central",
        },
        {
            "id": 3,
            "provider_team_id": 103,
            "name": "Metro Comets",
            "abbreviation": "MTC",
            "city": "Metro",
            "conference": "Oeste",
            "division": "Pacífica",
        },
        {
            "id": 4,
            "provider_team_id": 104,
            "name": "River Kings",
            "abbreviation": "RVK",
            "city": "River",
            "conference": "Oeste",
            "division": "Noroeste",
        },
    ]
    ids = [1, 2, 3, 4]
    pairs = [(home, away) for home in ids for away in ids if home != away]
    games: list[dict] = []
    game_id = 1
    for season in (2022, 2023, 2024):
        start = date(season, 10, 18)
        schedule = pairs + pairs
        for index, (home, away) in enumerate(schedule):
            game_date = start + timedelta(days=index * 2)
            home_score = 100 + ((home * 7 + away * 3 + season + index) % 18)
            away_score = 96 + ((away * 5 + home + index * 2) % 17)
            if home_score == away_score:
                home_score += 1
            games.append(
                {
                    "id": game_id,
                    "provider": "synthetic",
                    "source_game_id": 5000 + game_id,
                    "season": season,
                    "game_date": game_date.isoformat(),
                    "status": "final",
                    "home_team_id": home,
                    "visitor_team_id": away,
                    "home_score": home_score,
                    "visitor_score": away_score,
                    "postseason": False,
                }
            )
            game_id += 1
    games.append(
        {
            "id": game_id,
            "provider": "synthetic",
            "source_game_id": 5000 + game_id,
            "season": 2025,
            "game_date": "2025-10-22",
            "status": "scheduled",
            "home_team_id": 1,
            "visitor_team_id": 2,
            "home_score": None,
            "visitor_score": None,
            "postseason": False,
        }
    )
    return {
        "disclaimer": "Fixture sintético. Não representa partidas, marcas ou estatísticas oficiais.",
        "teams": teams,
        "games": games,
    }
