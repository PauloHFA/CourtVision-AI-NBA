from __future__ import annotations

from datetime import date

from sqlalchemy import or_, select
from sqlalchemy.sql import Select

from courtvision.data.models import Game

# Consulta de referência (apêndice D). Os binds são parâmetros, nunca texto do usuário.
PRIOR_GAMES_SQL = """
SELECT g.source_game_id, g.game_date,
       CASE WHEN g.home_team_id = :team_id
            THEN g.home_score ELSE g.visitor_score END AS points_for,
       CASE WHEN g.home_team_id = :team_id
            THEN g.visitor_score ELSE g.home_score END AS points_against
FROM games g
WHERE g.status = 'final'
  AND g.game_date < :cutoff_date
  AND (g.home_team_id = :team_id OR g.visitor_team_id = :team_id)
ORDER BY g.game_date DESC, g.source_game_id DESC
LIMIT 10
"""


def select_prior_games(team_id: int, cutoff_date: date, limit: int = 10) -> Select:
    return (
        select(Game)
        .where(Game.status == "final")
        .where(Game.game_date < cutoff_date)
        .where(or_(Game.home_team_id == team_id, Game.visitor_team_id == team_id))
        .order_by(Game.game_date.desc(), Game.source_game_id.desc())
        .limit(limit)
    )
