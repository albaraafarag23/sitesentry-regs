# SiteSentry Regs

A bilingual (Arabic/English) assistant that answers questions about UAE construction-safety
regulations and cites the exact clause it used.

It is the companion to [SiteSentry](https://github.com/albaraafarag23), an edge-AI
construction-site safety system: SiteSentry detects a violation on site, and this
assistant says which rule it breaks.

> Status: work in progress. Weekend 1 of 3 (RAG with cited answers) is done.

## How it works

```
PDF -> pages -> clauses (chunks) -> embeddings -> PostgreSQL + pgvector      (ingestion, run once)

question -> embedding -> nearest clauses -> Claude writes answer with [n] citations   (every request)
```

- **Chunking** (`app/chunking.py`): each PDF is split at its numbered headings, so every
  chunk knows its clause number, heading and page.
- **Embeddings** (`app/embeddings.py`): `intfloat/multilingual-e5-small` from Hugging Face
  turns text into 384-number vectors. It is multilingual, so an Arabic question can match
  an English clause.
- **Vector search** (`app/db.py`): chunks live in PostgreSQL with the pgvector extension;
  the nearest chunks are found with cosine distance and an HNSW index.
- **Answering** (`app/answer.py`): Claude gets the retrieved clauses as numbered sources
  and must cite them. The API maps the numbers back to document, clause and page.

## Documents

Official codes of practice from the Sharjah Prevention and Safety Authority (OSHJ):

| Code | Title | Languages |
|---|---|---|
| OSHJ-CoP-04 | Working at Height Safety | English, Arabic |
| OSHJ-CoP-02 | OSH Management in Construction | English |

## Run it

Requirements: Docker Desktop, Python 3.12+.

```bash
docker compose up -d                      # PostgreSQL + pgvector on port 5433
python -m venv .venv
.venv/Scripts/activate                    # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # then put your ANTHROPIC_API_KEY in .env
python -m app.download                    # fetch the official PDFs into data/pdfs
python -m app.ingest                      # chunk, embed and store them
uvicorn app.main:app --reload
```

Ask a question:

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "When must a ladder be inspected?"}'
```

`POST /search` takes the same body and returns only the retrieved clauses (no LLM call).
Interactive docs are at http://localhost:8000/docs.

## Tests

```bash
pytest                 # chunking tests always run; retrieval tests need the database
```

## Roadmap

- [x] Weekend 1: RAG with cited answers (FastAPI, pgvector, Hugging Face embeddings)
- [ ] Weekend 2: agent with tools and conversation memory, the same tools as an MCP server
- [ ] Weekend 3: evaluation set with a measured accuracy figure, CI, cloud deployment, demo
