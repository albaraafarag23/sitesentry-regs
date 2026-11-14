"""Small language helpers (no heavy imports, so the slim server can use them)."""

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}


def detect_language(text: str) -> str:
    """Return "ar" if most letters are Arabic, otherwise "en"."""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return "ar" if arabic > latin else "en"


def language_instruction(message: str) -> str:
    """One explicit line for the system prompt saying which language to answer in.

    Search returns clauses in both languages, and a general "answer in the user's
    language" rule was not enough: an English question once got an Arabic answer
    because Arabic sources ranked first. Detecting the language in code and
    stating it removes the guesswork.
    """
    name = LANGUAGE_NAMES[detect_language(message)]
    return f"The user's message is in {name}. Write your whole answer in {name}, whatever language the sources are in."
