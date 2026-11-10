from __future__ import annotations

FEATURE_VERSION = "features-1"
NEUTRAL_WIN_RATE = 0.5
NEUTRAL_POINTS = 110.0
NEUTRAL_REST_DAYS = 7.0
INITIAL_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
FINAL_STATUS = "final"

FEATURE_COLUMNS = [
    "home_win_rate_at_home",
    "away_win_rate_on_road",
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_win_rate_last_10",
    "away_win_rate_last_10",
    "home_current_win_streak",
    "away_current_win_streak",
    "home_points_for_avg_5",
    "away_points_for_avg_5",
    "home_points_for_avg_10",
    "away_points_for_avg_10",
    "home_points_against_avg_5",
    "away_points_against_avg_5",
    "home_points_against_avg_10",
    "away_points_against_avg_10",
    "diff_win_rate_last_10",
    "diff_point_margin_last_10",
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
    "home_days_since_last_game",
    "away_days_since_last_game",
    "home_back_to_back",
    "away_back_to_back",
    "home_win_rate",
    "away_win_rate",
    "season_id",
    "home_game_number_in_season",
    "away_game_number_in_season",
    "home_imputed",
    "away_imputed",
    "home_short_history",
    "away_short_history",
    "min_prior_games",
]

DISCLAIMER = (
    "Estimativa estatística educacional; não garante resultado e não é aconselhamento de apostas."
)
