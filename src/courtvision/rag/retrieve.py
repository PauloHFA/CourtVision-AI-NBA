from __future__ import annotations

import re
from datetime import datetime

from courtvision.rag.documents import EvidenceDocument, is_hostile


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9áéíóúãõç]{3,}", text.lower())}


def retrieve(
    documents: list[EvidenceDocument],
    query: str,
    as_of: datetime,
    team_ids: set[int] | None = None,
    k: int = 5,
) -> list[EvidenceDocument]:
    query_tokens = _tokens(query)
    eligible = []
    for document in documents:
        if document.available_at > as_of or document.hostile or is_hostile(document.text):
            continue
        if team_ids and document.team_ids and not (set(document.team_ids) & team_ids):
            if document.doc_type != "methodology":
                continue
        score = len(query_tokens & _tokens(document.text))
        eligible.append((score, document))
    eligible.sort(key=lambda item: item[0], reverse=True)
    ranked = [document for score, document in eligible if score > 0]
    if not ranked:
        ranked = [document for _score, document in eligible if document.doc_type == "methodology"]
    return ranked[:k]
