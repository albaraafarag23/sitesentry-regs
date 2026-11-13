-- Turn on pgvector: adds the "vector" column type and distance operators.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row = one chunk of a regulation, with its clause number and its embedding.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document    TEXT NOT NULL,      -- e.g. "OSHJ-CoP-04"
    doc_title   TEXT NOT NULL,      -- e.g. "Working at Height Safety"
    language    TEXT NOT NULL,      -- "en" or "ar"
    clause      TEXT NOT NULL,      -- e.g. "4.3.1"
    heading     TEXT NOT NULL,      -- the clause heading
    page        INT  NOT NULL,      -- page where the clause starts
    content     TEXT NOT NULL,      -- the chunk text
    embedding   vector(384) NOT NULL  -- 384 = output size of multilingual-e5-small
);

-- HNSW index: finds nearest vectors fast without comparing against every row.
-- vector_cosine_ops = build it for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
