"""The same agent, built with LangChain / LangGraph instead of a hand-written loop.

app/agent.py writes the tool-calling loop by hand, to show how an agent works.
This file gets the same behaviour from a framework:

  - `@tool` turns a Python function into a tool; LangChain builds the JSON schema
    the model needs from the type hints and the docstring.
  - `create_agent` returns a LangGraph graph that runs the loop for us:
    model -> tool calls -> tool results -> model -> ... -> final answer.

Both versions share the tools (app/tools.py), the system prompt, the source
numbering and the citation check, so their answers can be compared fairly.
Choose one per request with "engine": "native" or "langchain" on POST /chat.
"""

import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent import MAX_STEPS, SYSTEM_PROMPT, AgentResult, SourceRegistry, run_tool
from app.answer import LLMError, cited_numbers
from app.config import GEMINI_MODEL, TOP_K
from app.lang import language_instruction


def build_tools(registry: SourceRegistry, steps: list[dict]) -> list:
    """Create the tools for one request.

    They are created per request because each one writes into this request's
    source registry (for citation numbers) and step list (for transparency).
    """

    def call(name: str, arguments: dict) -> dict:
        arguments = {key: value for key, value in arguments.items() if value is not None}
        steps.append({"tool": name, "arguments": arguments})
        return run_tool(name, arguments, registry)

    @tool
    def search_regulations(
        query: str,
        language: str | None = None,
        document: str | None = None,
        top_k: int = TOP_K,
    ) -> dict:
        """Search the regulations by meaning (Arabic or English). Returns the most relevant clauses.

        Args:
            query: What to look for, e.g. "guard rail height".
            language: "en" or "ar" to search only that language. Omit to search both.
            document: Only search this document code, e.g. "OSHJ-CoP-04".
            top_k: How many clauses to return (1-10).
        """
        return call("search_regulations", {"query": query, "language": language, "document": document, "top_k": top_k})

    @tool
    def get_clause(document: str, clause: str, language: str = "en") -> dict:
        """Fetch the full text of one clause when you know its number.

        Args:
            document: Document code, e.g. "OSHJ-CoP-04".
            clause: Clause number, e.g. "5.4.1".
            language: "en" or "ar".
        """
        return call("get_clause", {"document": document, "clause": clause, "language": language})

    @tool
    def list_documents() -> dict:
        """List the regulation documents available and their languages."""
        return call("list_documents", {})

    return [search_regulations, get_clause, list_documents]


def message_text(message) -> str:
    """Get plain text from a LangChain message (content can be a string or a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def run_agent_langchain(message: str, history: list[dict] | None = None) -> AgentResult:
    """Answer one user message with the LangChain agent. Same inputs and outputs as agent.run_agent."""
    registry = SourceRegistry()
    steps: list[dict] = []

    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
        timeout=60,
        max_retries=3,          # also retries when the free tier says "too many requests"
    )
    agent = create_agent(
        model,
        tools=build_tools(registry, steps),
        system_prompt=f"{SYSTEM_PROMPT}\n- {language_instruction(message)}",
    )

    messages = [{"role": turn["role"], "content": turn["content"]} for turn in history or []]
    messages.append({"role": "user", "content": message})

    try:
        # recursion_limit caps the number of graph steps (each model call and each tool run is one).
        state = agent.invoke({"messages": messages}, config={"recursion_limit": 2 * MAX_STEPS + 1})
    except Exception as error:
        raise LLMError(f"LangChain agent error: {type(error).__name__}: {error}") from error

    answer = message_text(state["messages"][-1]).strip()
    if not answer:
        raise LLMError("The LangChain agent returned no text.")
    return AgentResult(answer, registry.sources, cited_numbers(answer, len(registry.sources)), steps)
