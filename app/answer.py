"""Ask Claude to answer a question using only the retrieved clauses.

This is the "generation" half of RAG (Retrieval-Augmented Generation):
  1. retrieval found the most relevant clauses (app/retrieval.py)
  2. we paste those clauses into the prompt as numbered sources
  3. the model must answer from them and mark each claim with [1], [2], ...
  4. we turn those numbers back into real clause references

Because citations are numbers that point at chunks we retrieved ourselves,
the model cannot invent a clause number that does not exist.
"""

import re

import anthropic

from app.config import ANTHROPIC_MODEL

SYSTEM_PROMPT = """You answer questions about UAE construction-safety regulations.

You are given numbered source clauses taken from official codes of practice. Rules:
- Use only the sources. Do not add requirements from general knowledge.
- After every statement, cite the source it comes from with its number in square brackets, like [1] or [2][3].
- If the sources do not answer the question, say so plainly and do not guess.
- Answer in the language of the question (Arabic or English), even if the sources are in the other language.
- Be short and practical: a site engineer should be able to act on the answer."""


def format_sources(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered sources for the prompt."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{number}] {chunk['document']} ({chunk['doc_title']}), "
            f"clause {chunk['clause']} \"{chunk['heading']}\", page {chunk['page']}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(blocks)


def cited_numbers(answer: str, source_count: int) -> list[int]:
    """Find which source numbers the answer cites, in order, without repeats."""
    seen: list[int] = []
    for match in re.findall(r"\[(\d+)\]", answer):
        number = int(match)
        if 1 <= number <= source_count and number not in seen:
            seen.append(number)
    return seen


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call Claude with the question and the numbered sources; return its text."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.beta.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        # If the model declines a request, let the API retry it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}",
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this question.")
    # The response is a list of blocks (thinking, text, ...). Keep only the text.
    return "".join(block.text for block in response.content if block.type == "text").strip()
