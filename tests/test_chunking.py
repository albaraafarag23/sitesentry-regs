"""Tests for chunking. These need no database, no model and no PDF."""

from app.answer import cited_numbers
from app.chunking import chunk_document, clean_pages, follows, parse_clause_number, split_long
from app.pdf_text import detect_language, fix_arabic

HEADER = "Working at Height Safety\nPage {n} of 3\nOSHJ-CoP-04\n"

PAGES = [
    HEADER.format(n=1) + "Table of Contents\n1\nIntroduction ........ 2\n2\nRequirements ........ 2\n2.1\nLadders ........ 3\n",
    HEADER.format(n=2) + "1\nIntroduction\nFalls from height cause injuries.\n2\nRequirements\n2.1\nLadders\nLadders shall be inspected.\n",
    HEADER.format(n=3) + "They shall be secured.\n\nAt the top;\n\nAt the base.\n2.2 Guard Rails\nGuard rails shall be 950 mm high.\n",
]


def test_each_clause_becomes_a_chunk_with_its_number():
    chunks = chunk_document(PAGES)
    assert [c.clause for c in chunks] == ["1", "2.1", "2.2"]
    assert [c.heading for c in chunks] == ["Introduction", "Ladders", "Guard Rails"]


def test_chunk_keeps_page_and_heading_path():
    ladders = chunk_document(PAGES)[1]
    assert ladders.page == 2
    assert ladders.path == "Requirements > Ladders"


def test_clause_text_continues_across_pages_and_keeps_bullets():
    ladders = chunk_document(PAGES)[1]
    assert ladders.content == "Ladders shall be inspected. They shall be secured. - At the top; - At the base."


def test_headers_page_numbers_and_contents_page_are_removed():
    cleaned = clean_pages(PAGES)
    assert cleaned[0] == []                                   # table of contents page
    text = " ".join(line for lines in cleaned for line in lines)
    assert "Page" not in text and "OSHJ-CoP-04" not in text


def test_stray_number_is_not_a_clause():
    pages = ["1\nScope\nThe limit is\n7\nmetres above ground.\n"]
    chunks = chunk_document(pages)
    assert [c.clause for c in chunks] == ["1"]
    assert "7 metres" in chunks[0].content


def test_clause_sequence_rules():
    assert follows(None, [1])
    assert follows([5, 4, 1], [5, 4, 2])      # next sibling
    assert follows([5, 4], [5, 4, 1])         # first child
    assert follows([5, 4, 3], [5, 5])         # back up one level
    assert follows([5, 6], [6])               # back to top level
    assert not follows([5, 4], [5, 6])        # skips 5.5
    assert not follows([5, 4], [7, 1])        # wrong parent


def test_arabic_reversed_clause_number_is_read_correctly():
    # A right-to-left PDF gives "1.4" for clause 4.1.
    assert parse_clause_number("1.4", previous=[4]) == [4, 1]
    pages = ["1\nمقدمة\nنص المقدمة.\n2\nالمتطلبات\n1.2\nالسلالم\nيجب فحص السلالم.\n"]
    chunks = chunk_document(pages)
    assert [c.clause for c in chunks] == ["1", "2.1"]
    assert chunks[1].path == "المتطلبات > السلالم"


def test_long_clause_is_split_but_keeps_its_number():
    body = "\n".join(f"Sentence number {i} about scaffolds." for i in range(40))
    chunks = chunk_document([f"1\nScaffolds\n{body}\n"], max_chars=300)
    assert len(chunks) > 1
    assert {c.clause for c in chunks} == {"1"}
    assert all(len(c.content) <= 300 for c in chunks)


def test_split_long_cuts_only_between_lines():
    assert split_long(["aaaa", "bbbb", "cccc"], max_chars=9) == ["aaaa bbbb", "cccc"]


def test_arabic_ligature_repair():
    assert fix_arabic("األماكن املرتفعة") == "الأماكن المرتفعة"
    assert fix_arabic("العامل") == "العامل"          # a correct word is left alone


def test_language_detection():
    assert detect_language("Ladders shall be inspected") == "en"
    assert detect_language("يجب فحص السلالم OSHJ") == "ar"


def test_cited_numbers_are_unique_in_order_and_in_range():
    assert cited_numbers("Use a harness [2]. Inspect it [1][2]. See [9].", source_count=3) == [2, 1]
