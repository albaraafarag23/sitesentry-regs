"""FastAPI service: ask a question, get an answer with clause citations."""

import os

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.answer import cited_numbers, generate_answer
from app.retrieval import retrieve

app = FastAPI(title="SiteSentry Regs", description="Cited answers about UAE construction-safety regulations.")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["When is a guardrail required?"])


class Source(BaseModel):
    number: int          # the [n] used in the answer
    document: str        # e.g. "OSHJ-CoP-04"
    doc_title: str
    language: str        # "en" or "ar"
    clause: str          # e.g. "4.3.1"
    heading: str
    page: int
    similarity: float    # 1.0 = identical meaning, 0 = unrelated
    content: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Source]   # only the sources the answer actually cites
    retrieved: list[Source]   # everything retrieval found, for transparency


def to_sources(chunks: list[dict]) -> list[Source]:
    return [
        Source(number=number, **{**chunk, "similarity": round(float(chunk["similarity"]), 4)})
        for number, chunk in enumerate(chunks, start=1)
    ]


@app.post("/search", response_model=list[Source])
def search(body: Question) -> list[Source]:
    """Retrieval only, no LLM. Useful to see what the search finds."""
    return to_sources(retrieve(body.question))


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:
    """Retrieve relevant clauses, then have Claude answer from them with citations."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set. Add it to .env, or use /search.")

    chunks = retrieve(body.question)
    if not chunks:
        raise HTTPException(404, "No regulations ingested yet. Run: python -m app.ingest")

    try:
        text = generate_answer(body.question, chunks)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "The Claude API rejected the key in .env.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Claude API rate limit reached. Try again shortly.")
    except (anthropic.APIError, RuntimeError) as error:
        raise HTTPException(502, f"Claude API error: {error}")

    sources = to_sources(chunks)
    cited = cited_numbers(text, len(sources))
    return Answer(
        question=body.question,
        answer=text,
        citations=[sources[n - 1] for n in cited],
        retrieved=sources,
    )
