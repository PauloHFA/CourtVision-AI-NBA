from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from courtvision.errors import PermanentAuthError, PermanentRequestError, TemporaryApiError


class BallDontLieClient:
    """Cliente paginado. Não contorna 401/403 e não registra a chave."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.balldontlie.io/v1",
        transport: httpx.BaseTransport | None = None,
        min_interval_s: float = 15.0,
        max_retries: int = 5,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        if not api_key:
            raise PermanentAuthError("BDL_API_KEY ausente. A ingestão foi interrompida.")
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._sleeper = sleeper
        self._clock = clock
        self._rng = rng or random.Random(0)
        self._last_request = 0.0
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": api_key},
            timeout=30.0,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BallDontLieClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def iter_games(self, season: int, cursor: int | None = None) -> Iterator[dict[str, Any]]:
        for page, _next in self.iter_game_pages(season, cursor=cursor):
            yield from page

    def iter_game_pages(
        self,
        season: int,
        cursor: int | None = None,
    ) -> Iterator[tuple[list[dict[str, Any]], int | None]]:
        current = cursor
        while True:
            params: dict[str, Any] = {"seasons[]": season, "per_page": 100}
            if current is not None:
                params["cursor"] = current
            payload = self._get("/games", params)
            page = list(payload.get("data") or [])
            next_cursor = payload.get("meta", {}).get("next_cursor")
            yield page, next_cursor
            if next_cursor is None:
                break
            current = next_cursor

    def iter_teams(self) -> Iterator[dict[str, Any]]:
        cursor = None
        while True:
            params: dict[str, Any] = {"per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor
            payload = self._get("/teams", params)
            yield from payload.get("data") or []
            cursor = payload.get("meta", {}).get("next_cursor")
            if cursor is None:
                break

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = self._clock()
        wait = self.min_interval_s - (now - self._last_request)
        if wait > 0 and self._last_request:
            self._sleeper(wait)
        self._last_request = self._clock()

    def _backoff(self, attempt: int) -> float:
        return min(30.0, float(2 ** (attempt - 1)) + self._rng.random())

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self._client.get(path, params=params)
            except httpx.TimeoutException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise TemporaryApiError("Timeout ao consultar a fonte.") from exc
                self._sleeper(self._backoff(attempt))
                continue
            status = response.status_code
            if status in (401, 403):
                raise PermanentAuthError(
                    f"Autenticação recusada pela fonte ({status}). A restrição não será contornada."
                )
            if status == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise TemporaryApiError("Limite da fonte excedido após retentativas.")
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else self._backoff(attempt)
                self._sleeper(delay)
                continue
            if status >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise TemporaryApiError(f"Fonte indisponível ({status}).")
                self._sleeper(self._backoff(attempt))
                continue
            if status >= 400:
                raise PermanentRequestError(f"Requisição rejeitada pela fonte ({status}).")
            body = response.json()
            if not isinstance(body, dict):
                raise PermanentRequestError("Resposta da fonte não é um objeto JSON.")
            return body
