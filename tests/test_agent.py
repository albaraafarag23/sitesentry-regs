"""Tests for the agent's own logic. No LLM call and no database are needed:
the tool functions are replaced with fakes."""

from app import agent
from app.agent import SourceRegistry, for_memory, run_tool

LADDERS = {"id": 11, "document": "OSHJ-CoP-04", "doc_title": "Working at Height Safety", "language": "en",
           "clause": "5.4.1", "heading": "Ladders", "page": 7, "content": "Ladders shall be secured."}
RAILS = {"id": 22, "document": "OSHJ-CoP-04", "doc_title": "Working at Height Safety", "language": "en",
         "clause": "5.5.3", "heading": "Guard Rail Systems", "page": 10, "content": "Guard rails shall be 950 mm."}


def test_registry_numbers_sources_in_order_seen():
    registry = SourceRegistry()
    assert registry.number(LADDERS) == 1
    assert registry.number(RAILS) == 2


def test_registry_gives_the_same_chunk_the_same_number():
    registry = SourceRegistry()
    registry.number(LADDERS)
    registry.number(RAILS)
    assert registry.number(LADDERS) == 1          # seen again in a later tool call
    assert len(registry.sources) == 2


def test_run_tool_returns_clauses_with_source_numbers(monkeypatch):
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_regulations", lambda query: [LADDERS, RAILS])
    result = run_tool("search_regulations", {"query": "ladders"}, SourceRegistry())
    assert [c["source"] for c in result["clauses"]] == [1, 2]
    assert result["clauses"][0]["clause"] == "5.4.1"
    assert result["clauses"][0]["text"] == "Ladders shall be secured."


def test_run_tool_reports_errors_to_the_model_instead_of_crashing(monkeypatch):
    def broken(query):
        raise ConnectionError("database is down")

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_regulations", broken)
    result = run_tool("search_regulations", {"query": "ladders"}, SourceRegistry())
    assert "database is down" in result["error"]


def test_run_tool_rejects_unknown_tool_and_bad_arguments():
    assert "Unknown tool" in run_tool("delete_everything", {}, SourceRegistry())["error"]
    assert "error" in run_tool("get_clause", {"nonsense": 1}, SourceRegistry())


def test_answer_saved_to_memory_has_no_numbers_but_lists_its_clauses():
    answer = "Secure the ladder [1]. Rails must be 950 mm [2][1]."
    assert for_memory(answer, [LADDERS, RAILS], cited=[1, 2]) == (
        "Secure the ladder. Rails must be 950 mm.\n\n"
        "(Sources used in this answer: OSHJ-CoP-04 clause 5.4.1; OSHJ-CoP-04 clause 5.5.3)"
    )


def test_answer_without_citations_is_saved_as_plain_text():
    assert for_memory("The documents do not cover this.", [LADDERS], cited=[]) == "The documents do not cover this."


def test_langchain_message_text_handles_strings_and_blocks():
    from types import SimpleNamespace

    from app.agent_langchain import message_text

    assert message_text(SimpleNamespace(content="plain")) == "plain"
    blocks = [{"type": "text", "text": "Use a "}, {"type": "thinking", "thinking": "..."}, {"type": "text", "text": "harness [1]."}]
    assert message_text(SimpleNamespace(content=blocks)) == "Use a harness [1]."


def test_langchain_tools_match_the_native_tools():
    from app.agent_langchain import build_tools

    names = {t.name for t in build_tools(SourceRegistry(), [])}
    assert names == set(agent.TOOL_FUNCTIONS)


def test_every_declared_tool_has_a_function():
    declared = {declaration.name for declaration in agent.TOOL_DECLARATIONS}
    assert declared == set(agent.TOOL_FUNCTIONS)


def test_answer_language_is_decided_in_code_from_the_question():
    from app.lang import language_instruction

    assert "in English" in language_instruction("SiteSentry flagged no_harness near a roof edge")
    assert "in Arabic" in language_instruction("متى يجب فحص معدات العمل في الأماكن المرتفعة؟")
    # A mostly Arabic question that mentions an English document code is still Arabic.
    assert "in Arabic" in language_instruction("ما هي متطلبات OSHJ-CoP-04 للسلالم؟")
