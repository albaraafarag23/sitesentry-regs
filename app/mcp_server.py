"""MCP server: exposes the regulation tools to any MCP client.

MCP (Model Context Protocol) is a standard way for AI apps to use external tools.
Our own agent (app/agent.py) calls the tools directly in Python. This server offers
the *same* functions over MCP, so Claude Desktop, Claude Code, Cursor or any other
MCP client can search the regulations without knowing anything about our code.

Run it (the client normally starts it for you, talking over stdin/stdout):

    python -m app.mcp_server
"""

from mcp.server.mcpserver import MCPServer

from app import tools
from app.config import TOP_K

server = MCPServer(
    "sitesentry-regs",
    log_level="WARNING",
    instructions=(
        "Tools for UAE construction-safety codes of practice (Sharjah OSHJ), in English and Arabic. "
        "Use search_regulations to find relevant clauses, get_clause for the full text of a clause, "
        "and cite document, clause number and page in your answer."
    ),
)


@server.tool()
def search_regulations(
    query: str,
    language: str | None = None,
    document: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Search the regulations by meaning (Arabic or English) and return the most relevant clauses.

    language: "en" or "ar" to search one language only. document: a code such as "OSHJ-CoP-04".
    """
    return tools.search_regulations(query, language, document, top_k)


@server.tool()
def get_clause(document: str, clause: str, language: str = "en") -> list[dict]:
    """Fetch the full text of one clause by number, e.g. document "OSHJ-CoP-04", clause "5.4.1"."""
    return tools.get_clause(document, clause, language)


@server.tool()
def list_documents() -> list[dict]:
    """List the regulation documents available and their languages."""
    return tools.list_documents()


if __name__ == "__main__":
    server.run("stdio")
