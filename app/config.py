"""All settings in one place, read from environment variables (or the .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()  # copies values from .env into os.environ, if the file exists

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://regs:regs@localhost:5433/regs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")

# How many chunks to retrieve for each question.
TOP_K = int(os.getenv("TOP_K", "5"))
