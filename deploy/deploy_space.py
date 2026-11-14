"""Deploy the API and chat page to a Hugging Face Space (free Docker hosting).

    python deploy/deploy_space.py

Reads from .env:
  HF_TOKEN            Hugging Face access token with "write" permission
  HF_USERNAME         your Hugging Face username
  CLOUD_DATABASE_URL  connection string of the cloud PostgreSQL (Neon) with pgvector
  GEMINI_API_KEY      stored in the Space as a private secret, never in its files

Before the first deployment, load the regulations into the cloud database once:

    DATABASE_URL=<your cloud connection string> python -m app.ingest
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = "sitesentry-regs"


def main() -> None:
    load_dotenv(ROOT / ".env")
    missing = [name for name in ("HF_TOKEN", "HF_USERNAME", "CLOUD_DATABASE_URL", "GEMINI_API_KEY") if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = f"{os.environ['HF_USERNAME']}/{SPACE_NAME}"

    # 1. Create the Space (does nothing if it already exists).
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    # 2. Secrets live in the Space settings, not in the uploaded files.
    api.add_space_secret(repo_id, "GEMINI_API_KEY", os.environ["GEMINI_API_KEY"])
    api.add_space_secret(repo_id, "DATABASE_URL", os.environ["CLOUD_DATABASE_URL"])
    api.add_space_variable(repo_id, "LLM_PROVIDER", "gemini")
    api.add_space_variable(repo_id, "GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))

    # 3. Upload only what the container needs. Hugging Face then builds the Dockerfile.
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=["Dockerfile", "requirements.txt", "app/**", "data/documents.json"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="Deploy from GitHub repository",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=ROOT / "deploy" / "space_readme.md",
        path_in_repo="README.md",          # the Space reads its settings from this file's header
        commit_message="Space configuration",
    )
    print(f"Deployed. Build progress and logs: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
