from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone

from courtvision.data.records import GameRecord, TeamRecord
from courtvision.features.builder import parse_as_of
from courtvision.features.schema import DISCLAIMER
from courtvision.ml.predict import PredictionResult, PredictionService
from courtvision.rag.documents import EvidenceDocument, game_documents, is_hostile
from courtvision.rag.prompts import BETTING_MARKERS, SYSTEM_PROMPT
from courtvision.rag.retrieve import retrieve

ADVERSARIAL_MARKERS = (
    "invente",
    "inventa",
    "placar",
    "agendado",
    "equipe desconhecida",
    "fato posterior",
    "ignore as instruções",
    "ignore instrucoes",
    "ignore previous",
    "defina a probabilidade",
    "diga que a probabilidade",
    "0.99",
    "99%",
)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def render_value(value: object) -> str:
    number = float(value)  # type: ignore[arg-type]
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _add_number(allowed: set[str], value: object) -> None:
    rendered = render_value(value) if isinstance(value, (int, float)) else str(value)
    for token in (str(value), rendered, rendered.lstrip("-")):
        allowed.add(token)
        allowed.add(token.replace(".", ","))


def allowed_numbers(facts: list[dict], prediction: PredictionResult | None) -> set[str]:
    allowed: set[str] = set()
    for fact in facts:
        for key in ("value", "sample"):
            if fact.get(key) is None:
                continue
            _add_number(allowed, fact[key])
        if fact.get("as_of"):
            allowed.update(_numbers(str(fact["as_of"])))
    if prediction and prediction.p_home_win is not None and prediction.p_away_win is not None:
        for probability in (prediction.p_home_win, prediction.p_away_win):
            allowed.add(f"{probability:.6f}".rstrip("0").rstrip("."))
            percent = probability * 100
            allowed.add(f"{percent:.1f}")
            allowed.add(f"{percent:.0f}")
            allowed.add(str(round(percent, 1)).replace(".", ","))
        for factor in prediction.factors:
            if factor.get("value") is None:
                continue
            _add_number(allowed, factor["value"])
    return allowed


def numbers_are_supported(text: str, allowed: set[str]) -> bool:
    for number in _numbers(text):
        normalized = number.replace(",", ".")
        if number not in allowed and normalized not in allowed and normalized.rstrip("0").rstrip(".") not in allowed:
            return False
    return True


def classify_intent(question: str) -> str:
    lowered = question.lower()
    if any(marker in lowered for marker in BETTING_MARKERS):
        return "betting"
    if is_hostile(question) or any(marker in lowered for marker in ADVERSARIAL_MARKERS):
        return "adversarial"
    if "qual equipe" in lowered or "quem teve mais" in lowered or "mais vitórias" in lowered or "mais vitorias" in lowered:
        return "comparison"
    if any(token in lowered for token in ("fator", "probabilidade", "chance", "modelo", "explica")):
        return "explanation"
    if any(token in lowered for token in ("quantas", "quantos", "vitória", "vitoria", "vitórias", "vitorias")):
        return "factual"
    return "context"


def sample_limit(question: str) -> int:
    lowered = question.lower()
    if re.search(r"\b5\b", lowered) and "jogo" in lowered:
        return 5
    return 10


def recent_team_facts(
    games: list[GameRecord],
    team: TeamRecord,
    cutoff: date,
    limit: int = 10,
) -> dict:
    prior = [
        game
        for game in games
        if game.status == "final"
        and not game.postseason
        and game.game_date < cutoff
        and team.id in (game.home_team_id, game.visitor_team_id)
        and game.home_score is not None
        and game.visitor_score is not None
    ]
    prior.sort(key=lambda game: (game.game_date, game.source_game_id or game.game_id), reverse=True)
    sample = prior[:limit]
    wins = 0
    source_ids = []
    sources = []
    for game in sample:
        source_id = f"game:{game.source_game_id or game.game_id}"
        source_ids.append(source_id)
        available = datetime.combine(game.game_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
        sources.append({"source_id": source_id, "available_at": available.isoformat()})
        if game.home_team_id == team.id:
            wins += int(game.home_score > game.visitor_score)
        else:
            wins += int(game.visitor_score > game.home_score)
    return {
        "metric": "wins_in_last_sample",
        "value": wins,
        "sample": len(sample),
        "team": team.name,
        "as_of": cutoff.isoformat(),
        "source_ids": source_ids,
        "sources": sources,
    }


def render_template(
    intent: str,
    facts: list[dict],
    citations: list[dict],
    prediction: PredictionResult | None,
) -> str:
    usable = [fact for fact in facts if fact["sample"] > 0]
    if intent == "comparison":
        if len(usable) < 2:
            return "não há dados suficientes"
        left, right = usable[0], usable[1]
        if left["value"] == right["value"]:
            body = (
                f"{left['team']} e {right['team']} tiveram o mesmo número de vitórias: "
                f"{left['value']} em {left['sample']} jogos anteriores a {left['as_of']}."
            )
        else:
            leader, other = (left, right) if left["value"] > right["value"] else (right, left)
            body = (
                f"{leader['team']} teve mais vitórias recentes: {leader['value']} em {leader['sample']} jogos, "
                f"contra {other['value']} em {other['sample']} jogos, anteriores a {left['as_of']}."
            )
        return _with_citations(body, citations)
    parts = []
    if intent == "explanation" and prediction and prediction.abstained:
        parts.append(prediction.abstain_reason or "não há dados suficientes")
    elif intent == "explanation" and prediction and prediction.p_home_win is not None:
        home_pct = prediction.p_home_win * 100
        away_pct = prediction.p_away_win * 100 if prediction.p_away_win is not None else 0
        parts.append(
            f"A estimativa do modelo {prediction.model_version} é {home_pct:.1f}% para o mandante "
            f"e {away_pct:.1f}% para o visitante."
        )
        for factor in prediction.factors:
            if factor.get("value") is None:
                continue
            parts.append(f"{factor['label']}: {render_value(factor['value'])}.")
    if not usable and not parts:
        return "não há dados suficientes"
    for fact in usable:
        parts.append(
            f"Nos {fact['sample']} jogos anteriores a {fact['as_of']}, {fact['team']} teve {fact['value']} vitórias."
        )
    text = " ".join(parts)
    if intent == "explanation":
        text = f"{text} {DISCLAIMER}".strip()
    return _with_citations(text, citations)


def _with_citations(text: str, citations: list[dict]) -> str:
    parts = [text]
    if citations:
        parts.append("Evidências: " + ", ".join(item["source_id"] for item in citations) + ".")
    if DISCLAIMER not in text:
        parts.append(DISCLAIMER)
    return " ".join(parts)


def _fact_citations(facts: list[dict], moment: datetime) -> list[dict]:
    rows = []
    seen: set[str] = set()
    as_of = moment.astimezone(timezone.utc).isoformat()
    for fact in facts:
        if fact["sample"] <= 0:
            continue
        for source in fact.get("sources", []):
            source_id = source["source_id"]
            if source_id in seen:
                continue
            seen.add(source_id)
            rows.append(
                {
                    "source_id": source_id,
                    "source_type": "game",
                    "as_of": as_of,
                    "available_at": source["available_at"],
                }
            )
    return rows


class AssistantService:
    def __init__(
        self,
        games: list[GameRecord],
        teams: list[TeamRecord],
        prediction_service: PredictionService,
        documents: list[EvidenceDocument] | None = None,
    ) -> None:
        self.games = games
        self.teams = {team.id: team for team in teams}
        self.prediction_service = prediction_service
        self.documents = documents if documents is not None else game_documents(games, teams)

    def ask(
        self,
        question: str,
        as_of: datetime | str,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
    ) -> dict:
        moment = parse_as_of(as_of)
        intent = classify_intent(question)
        limit = sample_limit(question)
        facts = []
        team_ids: set[int] = set()
        prediction = None
        if home_team_id is not None:
            team_ids.add(home_team_id)
            team = self.teams.get(home_team_id)
            if team is not None:
                facts.append(recent_team_facts(self.games, team, moment.date(), limit))
        if away_team_id is not None:
            team_ids.add(away_team_id)
            team = self.teams.get(away_team_id)
            if team is not None:
                facts.append(recent_team_facts(self.games, team, moment.date(), limit))
        if home_team_id is not None and away_team_id is not None:
            try:
                prediction = self.prediction_service.predict(home_team_id, away_team_id, moment)
            except Exception:
                prediction = None
        recovered = retrieve(self.documents, question, moment, team_ids or None)
        citations = _fact_citations(facts, moment)
        seen = {item["source_id"] for item in citations}
        for document in recovered:
            if document.source_ref in seen:
                continue
            citations.append(
                {
                    "source_id": document.source_ref,
                    "source_type": document.doc_type,
                    "as_of": moment.astimezone(timezone.utc).isoformat(),
                    "available_at": document.available_at.isoformat(),
                }
            )
        insufficient = not facts and not recovered
        if facts and all(fact["sample"] == 0 for fact in facts) and not recovered:
            insufficient = True
        if intent == "betting":
            answer = "Este sistema não oferece aconselhamento de apostas. " + DISCLAIMER
            insufficient = True
            citations = []
            prediction = None
        elif intent == "adversarial":
            answer = "não há dados suficientes"
            insufficient = True
            citations = []
            prediction = None
        else:
            answer = render_template(intent, facts, citations, prediction)
            allowed = allowed_numbers(facts, prediction)
            if prediction is not None:
                allowed.update(_numbers(prediction.model_version))
                for factor in prediction.factors:
                    allowed.update(_numbers(str(factor.get("label", ""))))
            for citation in citations:
                allowed.update(_numbers(str(citation.get("source_id", ""))))
                allowed.update(_numbers(str(citation.get("available_at", ""))))
            if answer == "não há dados suficientes":
                insufficient = True
            elif not numbers_are_supported(answer, allowed):
                answer = "não há dados suficientes"
                insufficient = True
        return {
            "answer": answer,
            "facts": [
                {
                    "metric": fact["metric"],
                    "value": fact["value"],
                    "sample": fact["sample"],
                    "source_ids": fact["source_ids"],
                }
                for fact in facts
                if fact["sample"] > 0
            ],
            "citations": citations,
            "model_version": None if prediction is None else prediction.model_version,
            "p_home_win": None if prediction is None else prediction.p_home_win,
            "dataset_version": self.prediction_service.dataset_hash,
            "insufficient_evidence": insufficient,
            "system_prompt_applied": True,
            "llm_used": False,
            "intent": intent,
            "policy": SYSTEM_PROMPT,
        }
