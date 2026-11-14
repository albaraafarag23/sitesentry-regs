"""All settings in one place, read from environment variables (or the .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()  # copies values from .env into os.environ, if the file exists

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://regs:regs@localhost:5433/regs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Which LLM writes the answer: "gemini" (free tier) or "claude".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
# The API key variable each provider needs.
LLM_KEY_NAME = {"gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(LLM_PROVIDER, "GEMINI_API_KEY")

# How many chunks to retrieve for each question.
TOP_K = int(os.getenv("TOP_K", "8"))

# How many earlier messages of a conversation the agent sees (its "memory window").
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
