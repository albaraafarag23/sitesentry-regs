# Slim image for the API and chat page. No PyTorch: the embedding model runs on
# ONNX Runtime, so the whole server fits in a free 512 MB cloud instance.
FROM python:3.12-slim

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 app
WORKDIR /srv
RUN chown app:app /srv
USER app
ENV PATH="/home/app/.local/bin:${PATH}" \
    HF_HOME=/srv/hf-cache \
    EMBEDDING_BACKEND=onnx \
    EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    ONNX_MODEL_FILE=onnx/model_qint8_avx512_vnni.onnx

COPY --chown=app:app requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

# Download the tokenizer and the quantized model at build time, so the container starts fast.
RUN python -c "from huggingface_hub import hf_hub_download as get; import os; \
get(os.environ['EMBEDDING_MODEL'], 'sentencepiece.bpe.model'); get(os.environ['EMBEDDING_MODEL'], os.environ['ONNX_MODEL_FILE'])"

COPY --chown=app:app app ./app
COPY --chown=app:app data/documents.json ./data/documents.json

# Cloud hosts choose the port through the PORT variable; locally it is 8000.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
