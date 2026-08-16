-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm extension for fuzzy text matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- DROP ALL TABLES (reverse dependency order)
-- ============================================================
DROP TABLE IF EXISTS movie_genres CASCADE;
DROP TABLE IF EXISTS movie_companies CASCADE;
DROP TABLE IF EXISTS movie_countries CASCADE;
DROP TABLE IF EXISTS movie_languages CASCADE;
DROP TABLE IF EXISTS genres CASCADE;
DROP TABLE IF EXISTS production_companies CASCADE;
DROP TABLE IF EXISTS production_countries CASCADE;
DROP TABLE IF EXISTS spoken_languages CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS movies_simple CASCADE;

-- ============================================================
-- MOVIES TABLE (processed data + vector embedding)
-- ============================================================
CREATE TABLE movies (
    id              INTEGER PRIMARY KEY,
    budget          BIGINT,
    homepage        TEXT,
    original_language TEXT,
    original_title  TEXT,
    overview        TEXT,
    popularity      REAL,
    release_date    DATE,
    revenue         BIGINT,
    runtime         REAL,
    status          TEXT,
    tagline         TEXT,
    title           TEXT NOT NULL,
    vote_average    REAL,
    vote_count      INTEGER,
    embedding vector(384)
);

-- ============================================================
-- LOOKUP TABLES
-- ============================================================
CREATE TABLE genres (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL
);

CREATE TABLE production_companies (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL
);

CREATE TABLE production_countries (
    iso_3166_1  TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE spoken_languages (
    iso_639_1   TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

-- ============================================================
-- JUNCTION TABLES (many-to-many)
-- ============================================================
CREATE TABLE movie_genres (
    movie_id    INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    genre_id    INTEGER REFERENCES genres(id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, genre_id)
);

CREATE TABLE movie_companies (
    movie_id    INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    company_id  INTEGER REFERENCES production_companies(id) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, company_id)
);

CREATE TABLE movie_countries (
    movie_id        INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    country_code    TEXT REFERENCES production_countries(iso_3166_1) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, country_code)
);

CREATE TABLE movie_languages (
    movie_id        INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    language_code   TEXT REFERENCES spoken_languages(iso_639_1) ON DELETE CASCADE,
    PRIMARY KEY (movie_id, language_code)
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_movies_release_date ON movies(release_date);
CREATE INDEX IF NOT EXISTS idx_movies_vote_average ON movies(vote_average);
CREATE INDEX IF NOT EXISTS idx_movies_runtime ON movies(runtime);
CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies(popularity);

-- HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_movies_embedding ON movies
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN trigram index for fuzzy title matching (pg_trgm)
CREATE INDEX IF NOT EXISTS idx_movies_title_trgm ON movies
    USING gin (original_title gin_trgm_ops);

-- ============================================================
-- CONVERSATION MEMORY (long-term)
-- ============================================================
-- CREATE TABLE IF NOT EXISTS conversations (
--     id              SERIAL PRIMARY KEY,
--     session_id      TEXT NOT NULL,
--     created_at      TIMESTAMP DEFAULT NOW(),
--     updated_at      TIMESTAMP DEFAULT NOW()
-- );

-- CREATE TABLE IF NOT EXISTS messages (
--     id              SERIAL PRIMARY KEY,
--     conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
--     role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
--     content         TEXT NOT NULL,
--     tool_name       TEXT,
--     tool_input      JSONB,
--     tool_output     JSONB,
--     created_at      TIMESTAMP DEFAULT NOW()
-- );

-- CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
-- CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);

-- ============================================================
-- SHORT-TERM MEMORY (result sets, active context per session)
-- ============================================================
-- CREATE TABLE IF NOT EXISTS session_state (
--     id              SERIAL PRIMARY KEY,
--     session_id      TEXT NOT NULL,
--     key             TEXT NOT NULL,
--     value           JSONB NOT NULL,
--     updated_at      TIMESTAMP DEFAULT NOW(),
--     UNIQUE(session_id, key)
-- );

CREATE INDEX IF NOT EXISTS idx_session_state_session ON session_state(session_id);
