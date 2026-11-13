"""Split a regulation into chunks, one per numbered clause.

Why not just cut the text every 500 characters? Because the answer must cite
the exact clause. So each chunk is one clause (for example 5.4.1 "Ladders"),
and it carries its clause number, heading and page with it into the database.

This module is pure Python (no PDF library, no database), so it is easy to test.
Input: a list of page texts. Output: a list of Chunk objects.
"""

import re
from collections import Counter
from dataclasses import dataclass

CLAUSE_NUMBER = re.compile(r"^\d+(\.\d+)*$")            # "5", "5.4", "5.4.1" alone on a line
INLINE_CLAUSE = re.compile(r"^(\d+(?:\.\d+)+)\s+(\S.{2,100})$")   # "5.5.5.1 Fall Arrest System"
PUNCTUATION_ONLY = re.compile(r"^[()\[\].,:;]+$")      # stray brackets left over from layout
PAGE_NUMBER = re.compile(r"^(page\s*\d+\s*of\s*\d+|صفحة\s*\d+)$", re.IGNORECASE)
BULLET_ONLY = re.compile(r"^[••\-–o]$")      # a bullet symbol alone on a line
MAX_CHARS = 1200   # the embedding model reads about 512 tokens; keep chunks below that


@dataclass
class Chunk:
    clause: str      # "5.4.1"
    heading: str     # "Ladders"
    path: str        # "Requirements > Types of Work at Height Equipment > Ladders"
    page: int        # page where the clause starts (1-based)
    content: str     # the text of the clause


def clean_pages(pages: list[str]) -> list[list[str]]:
    """Turn each page into a list of useful lines.

    Removes blank lines, page numbers, table-of-contents pages, and headers or
    footers (found automatically: any line repeated on more than half the pages).
    """
    pages_lines = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]

    # A line that appears on most pages is a header/footer, not content.
    seen_on = Counter(line for lines in pages_lines for line in set(lines))
    repeated = {
        line for line, count in seen_on.items()
        if len(pages) >= 3 and count > len(pages) / 2 and not CLAUSE_NUMBER.match(line) and not BULLET_ONLY.match(line)
    }

    cleaned = []
    for lines in pages_lines:
        is_contents_page = sum("...." in line for line in lines) >= 3
        if is_contents_page:
            cleaned.append([])
            continue
        cleaned.append([
            line for line in lines
            if line not in repeated and not PAGE_NUMBER.match(line) and not PUNCTUATION_ONLY.match(line)
        ])
    return cleaned


def follows(previous: list[int] | None, new: list[int]) -> bool:
    """Is `new` the next clause number after `previous`?

    After 5.4.1 the next clause can be 5.4.2 (next sibling), 5.4.1.1 (first child),
    5.5 or 6 (back up a level). A stray number in the text, like "2" in a table,
    does not fit the sequence, so it is not mistaken for a clause.
    """
    if previous is None:
        return new == [1]
    depth = len(new)
    if depth > len(previous) + 1 or new[:-1] != previous[: depth - 1]:
        return False
    if depth == len(previous) + 1:
        return new[-1] == 1                    # first child
    return new[-1] == previous[depth - 1] + 1  # next sibling at this depth


def parse_clause_number(text: str, previous: list[int] | None) -> list[int] | None:
    """If the text is the next clause number, return it as a list like [5, 4, 1].

    Arabic PDFs are right-to-left, so "4.1" is extracted as "1.4". We accept the
    number in either direction, whichever continues the sequence.
    """
    if not CLAUSE_NUMBER.match(text):
        return None
    parts = [int(p) for p in text.split(".")]
    for candidate in (parts, parts[::-1]):
        if follows(previous, candidate):
            return candidate
    return None


def find_clause_start(lines: list[str], i: int, previous: list[int] | None):
    """Does a new clause start at lines[i]?

    Returns (number, heading, lines_used) or None. Two layouts exist in these PDFs:
      number alone, heading on the next line:   "5.4.1" / "Ladders"
      number and heading on one line:           "5.5.5.1 Fall Arrest System"
    """
    line = lines[i]
    number = parse_clause_number(line, previous)
    if number and i + 1 < len(lines) and not CLAUSE_NUMBER.match(lines[i + 1]):
        return number, lines[i + 1], 2
    inline = INLINE_CLAUSE.match(line)
    if inline:
        number = parse_clause_number(inline.group(1), previous)
        if number:
            return number, inline.group(2), 1
    return None


def split_long(lines: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """Join lines into pieces of at most max_chars, cutting only between lines."""
    pieces, current = [], ""
    for line in lines:
        if current and len(current) + 1 + len(line) > max_chars:
            pieces.append(current)
            current = line
        else:
            current = f"{current} {line}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(pages: list[str], max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split a document (list of page texts) into clause chunks."""
    chunks: list[Chunk] = []
    headings: dict[int, str] = {}        # depth -> heading, to build the path
    number: list[int] | None = None      # clause we are currently inside
    start_page = 0
    body: list[str] = []
    bullet_next = False

    def close_clause() -> None:
        """Save the clause we just finished reading (if it has any text)."""
        if number is None or not body:
            return
        depth = len(number)
        path = " > ".join(headings[d] for d in range(1, depth + 1) if d in headings)
        for piece in split_long(body, max_chars):
            chunks.append(Chunk(".".join(map(str, number)), headings[depth], path, start_page, piece))

    for page_number, lines in enumerate(clean_pages(pages), start=1):
        i = 0
        while i < len(lines):
            line = lines[i]
            start = find_clause_start(lines, i, number)
            if start:
                close_clause()
                number, heading, lines_used = start
                start_page, body = page_number, []
                depth = len(number)
                headings[depth] = " ".join(heading.strip("() ").split())   # tidy spaces/brackets
                for deeper in [d for d in headings if d > depth]:
                    del headings[deeper]     # left that sub-section
                i += lines_used
                continue
            if BULLET_ONLY.match(line):
                bullet_next = True           # the bullet's text is on the next line
            elif number is not None:
                body.append(f"- {line}" if bullet_next else line)
                bullet_next = False
            i += 1

    close_clause()
    return chunks
