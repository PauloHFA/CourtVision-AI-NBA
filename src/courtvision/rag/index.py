from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from courtvision.config import get_settings
from courtvision.data.models import Base, Document, DocumentChunk
from courtvision.data.repository import MemoryRepository, SqlRepository
from courtvision.data.session import make_engine, make_session_factory
from courtvision.rag.documents import EvidenceDocument, game_documents

logger = logging.getLogger("courtvision.rag.index")


def embed_text(text: str, base_url: str, model: str, timeout: float = 3.0) -> list[float] | None:
    if not base_url or not model:
        return None
    url = f"{base_url.rstrip('/')}/api/embeddings"
    try:
        response = httpx.post(url, json={"model": model, "prompt": text}, timeout=timeout)
        if response.status_code == 200:
            payload = response.json()
            embedding = payload.get("embedding")
            if isinstance(embedding, list) and len(embedding) > 0:
                return [float(x) for x in embedding]
    except (httpx.HTTPError, ValueError):
        logger.debug("Falha ao gerar embedding para texto via Ollama: %s", text[:40])
    return None


def index_documents(
    session: Session,
    documents: Iterable[EvidenceDocument],
    *,
    embed_model: str = "",
    ollama_base_url: str = "",
) -> dict[str, int]:
    inserted_docs = 0
    updated_docs = 0
    embedded_chunks = 0

    existing_docs = {
        row.source_ref: row
        for row in session.scalars(select(Document)).all()
    }

    for doc in documents:
        existing = existing_docs.get(doc.source_ref)
        if existing is not None:
            if existing.checksum == doc.checksum:
                continue
            existing.text = doc.text
            existing.checksum = doc.checksum
            existing.available_at = doc.available_at
            existing.event_at = doc.event_at
            existing.metadata_json = {"team_ids": list(doc.team_ids), "hostile": doc.hostile}
            session.flush()

            # Atualizar ou recriar chunk
            chunk = session.scalar(
                select(DocumentChunk).where(DocumentChunk.document_id == existing.id)
            )
            embedding = embed_text(doc.text, ollama_base_url, embed_model) if embed_model else None
            if embedding is not None:
                embedded_chunks += 1
            if chunk is not None:
                chunk.content = doc.text
                chunk.metadata_json = {"team_ids": list(doc.team_ids)}
                if embedding is not None:
                    chunk.embedding_json = embedding
            else:
                session.add(
                    DocumentChunk(
                        document_id=existing.id,
                        chunk_index=0,
                        content=doc.text,
                        metadata_json={"team_ids": list(doc.team_ids)},
                        embedding_json=embedding,
                    )
                )
            updated_docs += 1
        else:
            new_doc = Document(
                doc_type=doc.doc_type,
                source_ref=doc.source_ref,
                event_at=doc.event_at,
                available_at=doc.available_at,
                text=doc.text,
                checksum=doc.checksum,
                metadata_json={"team_ids": list(doc.team_ids), "hostile": doc.hostile},
            )
            session.add(new_doc)
            session.flush()
            existing_docs[doc.source_ref] = new_doc

            embedding = embed_text(doc.text, ollama_base_url, embed_model) if embed_model else None
            if embedding is not None:
                embedded_chunks += 1
            session.add(
                DocumentChunk(
                    document_id=new_doc.id,
                    chunk_index=0,
                    content=doc.text,
                    metadata_json={"team_ids": list(doc.team_ids)},
                    embedding_json=embedding,
                )
            )
            inserted_docs += 1

    session.commit()
    return {
        "inserted": inserted_docs,
        "updated": updated_docs,
        "embedded": embedded_chunks,
        "total_documents": session.query(Document).count(),
        "total_chunks": session.query(DocumentChunk).count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa resumos de partidas e metodologia em documents / document_chunks.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--ollama-url", default=None)
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url or os.environ.get("DATABASE_URL", "")
    if not database_url:
        raise SystemExit("DATABASE_URL é obrigatório para indexar no banco de dados.")

    embed_model = args.embed_model or settings.ollama_embed_model
    ollama_url = args.ollama_url or settings.ollama_base_url

    engine = make_engine(database_url)
    Base.metadata.create_all(engine)
    factory = make_session_factory(database_url)

    with factory() as session:
        if args.fixture:
            payload = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
            repo = MemoryRepository.from_dataset(payload)
            games = repo.all_game_records()
            teams = repo.list_teams()
        else:
            sql_repo = SqlRepository(session)
            games = sql_repo.all_game_records()
            teams = sql_repo.list_teams()

        docs = game_documents(games, teams)
        stats = index_documents(
            session,
            docs,
            embed_model=embed_model,
            ollama_base_url=ollama_url,
        )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
