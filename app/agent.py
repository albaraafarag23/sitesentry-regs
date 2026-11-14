"""The agent: an LLM that decides for itself which tools to call, in a loop.

Difference from /ask (plain RAG):
  /ask   always does exactly one search, then answers.
  agent  reads the question and chooses: search once, search twice with different
         wording, fetch a whole clause by number, list the documents, or answer
         directly from the conversation so far.

The loop ("tool calling" / "function calling"):
  1. send the conversation + the tool descriptions to the model
  2. the model replies with either text (finished) or "call tool X with these arguments"
  3. we run the tool in Python and send the result back
  4. repeat until the model answers in text (or we hit MAX_STEPS)

The model never touches the database. It can only ask us to run one of our tools.
"""

import os
import re
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from app import tools
from app.answer import LLMError, cited_numbers, gemini_generate
from app.config import GEMINI_MODEL

MAX_STEPS = 6   # safety limit on model <-> tool round trips for one question

SYSTEM_PROMPT = """You are SiteSentry Regs, an assistant for UAE construction-safety regulations.

You have tools that look up official codes of practice. Rules:
- For any question about safety rules, call a tool first. Never answer rules from memory.
- One good search is usually enough. Search again only if the first results do not answer the question
  (rephrase, try the other language, or fetch a full clause). You can request several tool calls at once.
- The user may give a SiteSentry violation label such as no_helmet, no_harness, no_safety_vest or
  unguarded_edge. Search for the rule that the violation breaks.
- Every tool result has a "source" number. After each statement, cite its source like [1] or [2][3].
- Use only what the tools return. If they do not answer the question, say so plainly and do not guess.
- Answer in the language of the user's latest message (Arabic or English).
- Be short and practical: a site engineer should be able to act on the answer.
- Earlier answers in this conversation show their sources written out, like (OSHJ-CoP-04 clause 5.4.1).
  Use the conversation to understand follow-up questions such as "and for stepladders?", but in your
  new answer always cite with the bracketed source numbers from this turn's tool results, like [1]."""

# What the model is told about each tool: a name, a description and the arguments (JSON Schema).
# The model reads these descriptions to decide which tool to call, so they matter.
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_regulations",
        description="Search the regulations by meaning. Returns the most relevant clauses. "
                    "Works for Arabic and English queries.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'guard rail height'."},
                "language": {"type": "string", "enum": ["en", "ar"],
                             "description": "Only search this language. Omit to search both."},
                "document": {"type": "string",
                             "description": "Only search this document code, e.g. 'OSHJ-CoP-04'."},
                "top_k": {"type": "integer", "description": "How many clauses to return (1-10, default 5)."},
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="get_clause",
        description="Fetch the full text of one clause when you know its number.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "document": {"type": "string", "description": "Document code, e.g. 'OSHJ-CoP-04'."},
                "clause": {"type": "string", "description": "Clause number, e.g. '5.4.1'."},
                "language": {"type": "string", "enum": ["en", "ar"], "description": "Default 'en'."},
            },
            "required": ["document", "clause"],
        },
    ),
    types.FunctionDeclaration(
        name="list_documents",
        description="List the regulation documents available and their languages.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
]

TOOL_FUNCTIONS = {
    "search_regulations": tools.search_regulations,
    "get_clause": tools.get_clause,
    "list_documents": tools.list_documents,
}


class SourceRegistry:
    """Gives every chunk the agent sees one stable number for this request.

    The agent may call tools several times and get the same clause twice;
    it must still be [3] both times, so citations stay unambiguous.
    """

    def __init__(self) -> None:
        self.sources: list[dict] = []          # sources[0] is source number 1
        self._number_by_chunk_id: dict[int, int] = {}

    def number(self, chunk: dict) -> int:
        chunk_id = chunk["id"]
        if chunk_id not in self._number_by_chunk_id:
            self.sources.append(chunk)
            self._number_by_chunk_id[chunk_id] = len(self.sources)
        return self._number_by_chunk_id[chunk_id]


@dataclass
class AgentResult:
    answer: str
    sources: list[dict]                 # every chunk the agent looked at, in number order
    cited: list[int]                    # the source numbers used in the answer
    steps: list[dict] = field(default_factory=list)   # which tools were called, for transparency


def run_tool(name: str, arguments: dict, registry: SourceRegistry) -> dict:
    """Run one tool call requested by the model and shape the result for the model."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        result = function(**arguments)
    except Exception as error:               # bad arguments, database down, ...
        return {"error": f"{type(error).__name__}: {error}"}   # the model can read this and retry
    if name == "list_documents":
        return {"documents": result}
    return {
        "clauses": [
            {
                "source": registry.number(chunk),
                "document": chunk["document"],
                "language": chunk["language"],
                "clause": chunk["clause"],
                "heading": chunk["heading"],
                "page": chunk["page"],
                "text": chunk["content"],
            }
            for chunk in result
        ]
    }


def with_explicit_citations(answer: str, sources: list[dict]) -> str:
    """Rewrite [2] as (OSHJ-CoP-04 clause 5.4.1) before saving an answer to memory.

    Source numbers only mean something inside one request. Written out in full,
    the citation still makes sense when the model re-reads it in a later turn.
    """
    def replace(match: re.Match) -> str:
        number = int(match.group(1))
        if 1 <= number <= len(sources):
            source = sources[number - 1]
            return f"({source['document']} clause {source['clause']})"
        return match.group(0)

    return re.sub(r"\[(\d+)\]", replace, answer)


def run_agent(message: str, history: list[dict] | None = None) -> AgentResult:
    """Answer one user message. `history` is earlier turns: [{"role": "user"|"assistant", "content": str}]."""
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=60_000),
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
    )

    # Gemini calls the assistant role "model".
    contents = [
        types.Content(
            role="model" if turn["role"] == "assistant" else "user",
            parts=[types.Part(text=turn["content"])],
        )
        for turn in history or []
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    registry = SourceRegistry()
    steps: list[dict] = []

    for _ in range(MAX_STEPS):
        response = gemini_generate(client, model=GEMINI_MODEL, contents=contents, config=config)

        calls = response.function_calls
        if not calls:                                   # plain text: the agent is finished
            answer = (response.text or "").strip()
            if not answer:
                raise LLMError("Gemini returned no text.")
            return AgentResult(answer, registry.sources, cited_numbers(answer, len(registry.sources)), steps)

        # Keep the model's own "please call these tools" message in the conversation,
        # then add one result per call, so the next round sees what it asked for.
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in calls:
            arguments = dict(call.args or {})
            result = run_tool(call.name, arguments, registry)
            steps.append({"tool": call.name, "arguments": arguments})
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))
        contents.append(types.Content(role="user", parts=result_parts))

    raise LLMError(f"The agent did not finish within {MAX_STEPS} steps.")
