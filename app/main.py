"""FastAPI service: cited answers about UAE construction-safety regulations.

  POST /search  retrieval only (no LLM)
  POST /ask     plain RAG: one search, one answer
  POST /chat    agent: chooses its own tools, remembers the conversation
  GET  /        a small chat web page that uses /chat
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from app.agent import for_memory, run_agent
from app.answer import LLMError, cited_numbers, generate_answer
from app.config import HISTORY_LIMIT, LLM_KEY_NAME
from app.retrieval import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: make sure the chat-memory table exists."""
    try:
        db.ensure_schema()
    except Exception as error:   # database not up yet: the API still starts, endpoints will report it
        print(f"warning: could not prepare the database: {error}")
    yield


app = FastAPI(
    title="SiteSentry Regs",
    description="Cited answers about UAE construction-safety regulations.",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["When is a guardrail required?"])


class Source(BaseModel):
    number: int                       # the [n] used in the answer
    document: str                     # e.g. "OSHJ-CoP-04"
    doc_title: str
    language: str                     # "en" or "ar"
    clause: str                       # e.g. "5.4.1"
    heading: str
    page: int
    content: str
    similarity: float | None = None   # 1.0 = identical meaning; missing when fetched by clause number


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Source]   # only the sources the answer actually cites
    retrieved: list[Source]   # everything retrieval found, for transparency


class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=2000, examples=["How should a ladder be secured?"])
    session_id: str | None = Field(default=None, description="Send the id from the last reply to continue a conversation.")
    engine: Literal["native", "langchain"] = Field(
        default="native",
        description="native = hand-written tool loop (app/agent.py); langchain = LangChain/LangGraph agent.",
    )


class ToolStep(BaseModel):
    tool: str
    arguments: dict


class ChatReply(BaseModel):
    session_id: str
    engine: str
    answer: str
    citations: list[Source]
    steps: list[ToolStep]     # which tools the agent called, in order


def to_sources(chunks: list[dict]) -> list[Source]:
    return [Source(number=number, **chunk) for number, chunk in enumerate(chunks, start=1)]


def require_llm_key(name: str) -> None:
    if not os.getenv(name):
        raise HTTPException(503, f"{name} is not set. Add it to .env, or use /search.")


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    """The chat web page (a single static HTML file that calls POST /chat)."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health", include_in_schema=False)
def health() -> dict:
    """Cheap check used by the cloud host to see that the server is up."""
    return {"status": "ok"}


@app.post("/search", response_model=list[Source])
def search(body: Question) -> list[Source]:
    """Retrieval only, no LLM. Useful to see what the search finds."""
    return to_sources(retrieve(body.question))


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:
    """Plain RAG: retrieve relevant clauses, then have the LLM answer from them with citations."""
    require_llm_key(LLM_KEY_NAME)

    chunks = retrieve(body.question)
    if not chunks:
        raise HTTPException(404, "No regulations ingested yet. Run: python -m app.ingest")

    try:
        text = generate_answer(body.question, chunks)
    except LLMError as error:
        raise HTTPException(502, str(error))

    sources = to_sources(chunks)
    cited = cited_numbers(text, len(sources))
    return Answer(
        question=body.question,
        answer=text,
        citations=[sources[n - 1] for n in cited],
        retrieved=sources,
    )


@app.post("/chat", response_model=ChatReply)
def chat(body: ChatMessage) -> ChatReply:
    """Agent with memory: it picks its own tools and remembers earlier turns of the session."""
    require_llm_key("GEMINI_API_KEY")   # the agent uses Gemini function calling
    session_id = body.session_id or uuid.uuid4().hex

    with db.connect() as conn:
        history = db.get_history(conn, session_id, HISTORY_LIMIT)

    try:
        if body.engine == "langchain":
            from app.agent_langchain import run_agent_langchain   # imported here: LangChain is slow to load

            result = run_agent_langchain(body.message, history)
        else:
            result = run_agent(body.message, history)
    except LLMError as error:
        raise HTTPException(502, str(error))

    # Save both sides of this turn so the next request can see them.
    with db.connect() as conn:
        db.add_message(conn, session_id, "user", body.message)
        db.add_message(conn, session_id, "assistant", for_memory(result.answer, result.sources, result.cited))

    sources = to_sources(result.sources)
    return ChatReply(
        session_id=session_id,
        engine=body.engine,
        answer=result.answer,
        citations=[sources[n - 1] for n in result.cited],
        steps=[ToolStep(**step) for step in result.steps],
    )
