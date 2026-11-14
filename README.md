# SiteSentry Regs

A bilingual (Arabic/English) assistant that answers questions about UAE construction-safety
regulations and cites the exact clause it used.

It is the companion to SiteSentry, an edge-AI construction-site safety system: SiteSentry
detects a violation on site (for example `no_harness`), and this assistant says which rule
it breaks.

```
Q: How high must a guard rail be?
A: At least 950 mm, with an intermediate rail and gaps of no more than 470 mm [1].
   [1] OSHJ-CoP-04, clause 5.5.3 "Guard Rail Systems", page 10
```

## What is in it

| Part | What it does | Where |
|---|---|---|
| Clause-aware chunking | Splits each PDF at its numbered clauses, so every chunk keeps its clause number, heading and page. Handles right-to-left Arabic numbering. | `app/chunking.py` |
| Embeddings | `intfloat/multilingual-e5-small` (Hugging Face) turns text into 384-number vectors; Arabic and English share one vector space. | `app/embeddings.py` |
| ONNX inference | The deployed server runs the same model as an 8-bit quantized ONNX file with ONNX Runtime and SentencePiece, with no PyTorch, so it fits a free 512 MB instance. | `app/embeddings_onnx.py` |
| Vector search | PostgreSQL + pgvector, cosine distance, HNSW index. Each language is searched separately and the results are interleaved. | `app/db.py`, `app/tools.py` |
| RAG endpoint | One search, then the LLM answers from numbered sources and must cite them. | `app/answer.py` |
| Agent | The LLM chooses its own tools in a loop (search, fetch a clause, list documents) and remembers the conversation. | `app/agent.py` |
| LangChain agent | The same agent built with LangChain/LangGraph (`create_agent`, `@tool`), selectable per request, to compare a framework with the hand-written loop. | `app/agent_langchain.py` |
| Chat page | A single-file web page for `/chat`: Arabic right-to-left support, tool trace and expandable citations. | `app/static/index.html` |
| MCP server | The same tools exposed over the Model Context Protocol for Claude Desktop, IDEs and other MCP clients. | `app/mcp_server.py` |
| Evaluation | 47 labelled questions; reports hit@k and MRR for retrieval and citation accuracy for answers. | `app/evaluate.py`, `data/eval/` |
| Tests and CI | pytest (47 tests) and a GitHub Actions workflow with a pgvector service. | `tests/`, `.github/workflows/ci.yml` |

```
PDF -> pages -> clauses (chunks) -> embeddings -> PostgreSQL + pgvector        (ingestion, run once)

question -> embedding -> nearest clauses -> LLM answers with [n] citations     (/ask)
question -> agent <-> tools (search / get clause / list) -> cited answer        (/chat, with memory)
```

Citations cannot be invented: the LLM only cites source numbers, and the API maps each
number back to a chunk that the search itself returned.

## Documents

Official codes of practice from the Sharjah Prevention and Safety Authority (OSHJ).
The PDFs are not stored in this repository; `python -m app.download` fetches them from the source.

| Code | Title | Languages |
|---|---|---|
| OSHJ-CoP-04 | Working at Height Safety | English, Arabic |
| OSHJ-CoP-02 | OSH Management in Construction | English |

## Measured results

Measured with `python -m app.evaluate` on the 47 questions in `data/eval/questions.json`
(29 English, 10 Arabic, 4 Arabic questions about an English-only document, 4 that the
documents cannot answer). "hit@k" = the clause containing the answer is in the top k results.

**Retrieval**

| Questions | Count | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|---|
| All answerable | 43 | 84% | 95% | 95% | 98% | 0.90 |
| English | 29 | 97% | 97% | 97% | 100% | 0.97 |
| Arabic | 10 | 80% | 100% | 100% | 100% | 0.90 |
| Arabic question, English-only document | 4 | 0% | 75% | 75% | 75% | 0.38 |

**Answers** (plain RAG on `/ask`, model `gemini-3.5-flash-lite`, 8 retrieved clauses)

| Check | Result |
|---|---|
| Answerable questions where the answer cites a clause that contains the answer | 42 / 43 (98%) |
| Unanswerable questions where the answer cites nothing | 3 / 4 |

The one unanswerable miss asked for a noise limit in decibels; the answer cited the general
personal protective equipment clause instead of citing nothing.

**What the evaluation found.** The first version did one combined search and scored 0% on
the last row: the embedding model rates same-language text higher, so an Arabic question
only ever returned Arabic chunks. Searching each language separately and interleaving the
results raised that group to 75% and overall hit@3 from 86% to 95%
(before: `data/eval/results_retrieval_baseline.json`, after: `data/eval/results_retrieval.json`).

**PyTorch vs ONNX for the question embedding** (documents are always embedded with PyTorch)

| Question embedding | hit@1 | hit@3 | hit@5 | hit@8 | MRR | Process memory after one query |
|---|---|---|---|---|---|---|
| PyTorch, full precision | 84% | 95% | 95% | 98% | 0.90 | 1,298 MB |
| ONNX Runtime, 8-bit quantized | 84% | 93% | 95% | 98% | 0.89 | 259 MB |

The quantized vectors agree with the PyTorch ones to a cosine similarity of 0.997, and one
question embeds in a few milliseconds. Swapping the `tokenizers` library for the model's native
SentencePiece file cut the tokenizer's memory from about 270 MB to about 50 MB with identical token ids.

Limits of these numbers: the test set is small and was written by the project author, so it
is a regression check, not an independent benchmark.

## Run it

Requirements: Docker Desktop, Python 3.12+.

```bash
docker compose up -d                      # PostgreSQL + pgvector on port 5433
python -m venv .venv
.venv/Scripts/activate                    # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # then put your GEMINI_API_KEY in .env (free tier works)
python -m app.download                    # fetch the official PDFs into data/pdfs
python -m app.ingest                      # chunk, embed and store them
uvicorn app.main:app --reload
```

Chat page: http://localhost:8000/ and interactive API docs: http://localhost:8000/docs

| Endpoint | Purpose |
|---|---|
| `POST /search` | Retrieval only, no LLM. Body: `{"question": "..."}` |
| `POST /ask` | Plain RAG: one search, one cited answer. Body: `{"question": "..."}` |
| `POST /chat` | Agent with memory. Body: `{"message": "...", "session_id": "...", "engine": "native"}`; send back the `session_id` from the reply to continue the conversation. `engine` is `native` (hand-written loop) or `langchain`. |

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "SiteSentry flagged no_harness near a roof edge. Which rule does that break?"}'
```

### Use the tools from an MCP client

Add this to the MCP configuration of Claude Desktop, Claude Code or another client
(use the full path to this project's Python):

```json
{
  "mcpServers": {
    "sitesentry-regs": {
      "command": "C:/path/to/sitesentry-regs/.venv/Scripts/python.exe",
      "args": ["-m", "app.mcp_server"],
      "cwd": "C:/path/to/sitesentry-regs"
    }
  }
}
```

## Deployment

The API and chat page run as one small Docker container (`Dockerfile`, `requirements-serve.txt`)
next to a cloud PostgreSQL with pgvector:

- **Database:** Neon (free plan). Load it once from your machine:
  `DATABASE_URL=<neon connection string> python -m app.ingest`
- **Server:** Render (free plan), created from `render.yaml`. `GEMINI_API_KEY` and `DATABASE_URL`
  are entered as secrets in the Render dashboard.

Run the same container locally with `docker compose --profile full up`.

## Tests and evaluation

```bash
pytest                           # unit tests always run; database tests skip if the database is down
python -m app.evaluate           # retrieval metrics (free, about 10 seconds)
python -m app.evaluate --answers # also checks generated answers (uses the LLM)
```

## Known limitations

- The Arabic PDF stores some joined letters in the wrong order (for example السلامة is
  extracted as السالمة). Only the cases that are safe to repair are repaired.
- In the Arabic PDF, clause 5.5.5.1 is misplaced in the file, so 5.5.5.1 and 5.5.5.2 are
  stored under 5.5.5.
- The Gemini free tier is rate limited (a few requests per minute and a small daily quota
  per model), so `/chat` can be slow or temporarily unavailable.

## Roadmap

- [x] RAG with cited answers (FastAPI, pgvector, Hugging Face embeddings)
- [x] Agent with tools and conversation memory, the same tools as an MCP server
- [x] Evaluation set with measured retrieval figures, pytest, CI workflow, Dockerfile
- [x] ONNX Runtime inference and a slim server image
- [ ] Cloud deployment and a recorded demo
