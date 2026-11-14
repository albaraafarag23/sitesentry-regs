# Image for the API (FastAPI + the embedding model).
FROM python:3.12-slim

# Run as a normal user, not root. Hugging Face Spaces requires user id 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /srv
RUN chown app:app /srv
USER app
ENV PATH="/home/app/.local/bin:${PATH}" \
    HF_HOME=/srv/hf-cache

# CPU-only PyTorch first: the default build bundles GPU libraries and is several GB.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the embedding model at build time, so the container starts fast.
ENV EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY --chown=app:app app ./app
COPY --chown=app:app data/documents.json ./data/documents.json

# Cloud hosts choose the port through the PORT variable; locally it is 8000.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
