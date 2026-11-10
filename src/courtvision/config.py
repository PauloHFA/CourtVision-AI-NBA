from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bdl_api_key: str = ""
    database_url: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = ""
    ollama_embed_model: str = ""
    model_artifact_path: str = "models/champion.joblib"
    model_version: str = "ml-0.1.0"
    feature_version: str = "features-1"
    ingestion_rate_per_min: int = 4
    api_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    app_env: str = "development"
    fixture_path: str = "data/fixtures/synthetic_season.json"
    min_prior_games: int = 5
    api_rate_limit_per_min: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.api_allowed_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
