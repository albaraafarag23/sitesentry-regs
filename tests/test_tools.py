"""Tests for the tools, chat memory and the MCP server. These need the database
running with the PDFs ingested; otherwise they are skipped."""

import asyncio
import json
import uuid

import pytest

from app import db, tools

pytestmark = pytest.mark.usefixtures("ingested_database")


def test_search_can_be_limited_to_one_language():
    results = tools.search_regulations("ladder inspection", language="ar", top_k=4)
    assert len(results) == 4
    assert {r["language"] for r in results} == {"ar"}


def test_search_can_be_limited_to_one_document():
    results = tools.search_regulations("training for employees", document="OSHJ-CoP-02")
    assert {r["document"] for r in results} == {"OSHJ-CoP-02"}


def test_top_k_is_clamped_to_a_safe_range():
    assert len(tools.search_regulations("scaffold", top_k=500)) == tools.MAX_TOP_K
    assert len(tools.search_regulations("scaffold", top_k=0)) == 1


def test_get_clause_returns_all_pieces_of_the_clause_in_order():
    pieces = tools.get_clause("OSHJ-CoP-04", "5.4.1")
    assert len(pieces) >= 2                                   # a long clause is stored in pieces
    assert {p["clause"] for p in pieces} == {"5.4.1"}
    assert pieces[0]["heading"] == "Ladders"
    assert [p["id"] for p in pieces] == sorted(p["id"] for p in pieces)


def test_get_clause_unknown_clause_is_empty():
    assert tools.get_clause("OSHJ-CoP-04", "99.9") == []


def test_list_documents_shows_both_languages_of_cop_04():
    documents = tools.list_documents()
    languages = {d["language"] for d in documents if d["document"] == "OSHJ-CoP-04"}
    assert languages == {"en", "ar"}


def test_chat_memory_returns_the_latest_messages_oldest_first():
    session_id = f"test-{uuid.uuid4().hex}"
    with db.connect() as conn:
        for i in range(1, 6):
            db.add_message(conn, session_id, "user" if i % 2 else "assistant", f"message {i}")
    with db.connect() as conn:
        history = db.get_history(conn, session_id, limit=3)
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
    assert [m["content"] for m in history] == ["message 3", "message 4", "message 5"]
    assert history[0]["role"] == "user"


def test_mcp_server_exposes_the_same_tools_and_they_work():
    """Connect a real MCP client to our server (in memory) and call a tool through the protocol."""
    from mcp import Client

    from app.mcp_server import server

    async def scenario():
        async with Client(server) as client:
            listed = await client.list_tools()
            result = await client.call_tool("get_clause", {"document": "OSHJ-CoP-04", "clause": "5.5.3"})
            return [tool.name for tool in listed.tools], result

    names, result = asyncio.run(scenario())
    assert set(names) == {"search_regulations", "get_clause", "list_documents"}
    assert not result.is_error
    text = "".join(block.text for block in result.content if block.type == "text")
    assert "Guard Rail" in text
