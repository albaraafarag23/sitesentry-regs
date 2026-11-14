"""Read a PDF into a list of page texts, with light clean-up for Arabic."""

import re
from pathlib import Path

import pymupdf

from app.lang import detect_language  # noqa: F401  (re-exported for ingest and tests)

# In these Arabic PDFs some joined letter pairs (ligatures) are stored as one glyph,
# and text extraction returns the two letters in the wrong order:
#   "الأماكن" comes out as "األماكن",  "المرتفعة" comes out as "املرتفعة".
# We repair only the patterns that cannot be a real Arabic spelling.
_ARABIC_FIXES = [
    (re.compile("ا([أإآ])ل"), r"ال\1"),                       # األ -> الأ, اإل -> الإ, اآل -> الآ
    (re.compile(r"\b([وفبل]?)امل(?=\w\w)"), r"\1الم"),        # املرتفعة -> المرتفعة
]
# Known limitation: a plain "لا" inside a word is also swapped ("السلامة" -> "السالمة").
# That one cannot be repaired safely, because "ال" is also a correct, very common
# letter pair. The embedding model and the LLM both cope with it.


def fix_arabic(text: str) -> str:
    for pattern, replacement in _ARABIC_FIXES:
        text = pattern.sub(replacement, text)
    return text


def extract_pages(path: Path) -> list[str]:
    """Return the text of each page, in order. Index 0 is page 1."""
    with pymupdf.open(path) as doc:
        return [fix_arabic(page.get_text()) for page in doc]
