"""Ingestion: PDF -> clause chunks -> embeddings -> PostgreSQL (pgvector).

Run once, and again whenever the PDFs change:

    python -m app.ingest

The list of documents is in data/documents.json.
"""

import json
from pathlib import Path

from app import db
from app.chunking import chunk_document
from app.embeddings import embed_passages
from app.pdf_text import detect_language, extract_pages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def ingest_document(conn, entry: dict) -> int:
    """Chunk, embed and store one PDF. Returns the number of chunks stored."""
    pages = extract_pages(DATA_DIR / "pdfs" / entry["file"])
    language = detect_language(" ".join(pages))
    chunks = chunk_document(pages)

    # What we embed is the heading path plus the text, so that a chunk that only
    # says "it shall be inspected weekly" still knows it is about "Ladders".
    vectors = embed_passages([f"{chunk.path}\n{chunk.content}" for chunk in chunks])

    rows = [
        {
            "document": entry["document"],
            "doc_title": entry["title"],
            "language": language,
            "clause": chunk.clause,
            "heading": chunk.heading,
            "page": chunk.page,
            "content": chunk.content,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    db.replace_document(conn, entry["document"], language, rows)
    return len(rows)


def main() -> None:
    entries = json.loads((DATA_DIR / "documents.json").read_text(encoding="utf-8"))
    db.ensure_schema()
    with db.connect() as conn:
        for entry in entries:
            count = ingest_document(conn, entry)
            print(f"{entry['file']}: stored {count} chunks")


if __name__ == "__main__":
    main()
