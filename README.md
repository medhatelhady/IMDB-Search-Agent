# Movie Agent — AI-Powered Movie Discovery System

An agentic AI system for movie discovery and analysis built on the TMDB 5000 dataset. It combines structured SQL queries, semantic vector search, and fuzzy title matching through an intelligent multi-agent architecture powered by LangGraph.

## Features

- **Natural Language Queries** — Ask questions about movies in plain English
- **Multi-Agent Routing** — Automatically routes queries to the most appropriate agent:
  - **SQL Agent** — Structured/factual questions (counts, rankings, filtering by year/rating/genre)
  - **Semantic Search Agent** — Mood, theme, or plot-based movie recommendations using vector similarity
  - **Fuzzy Search Agent** — Title lookup with typo tolerance using trigram matching
- **Streaming Responses** — Real-time token streaming with agent step visibility
- **Session Memory** — Conversation continuity across multiple interactions
- **Dual Interface** — Streamlit chat UI and FastAPI REST API

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Database | PostgreSQL 16 + pgvector |
| Agent Framework | LangChain + LangGraph (ReAct pattern) |
| LLM | OpenAI GPT-4o-mini or Ollama (llama3.1:8b) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| Fuzzy Matching | pg_trgm (PostgreSQL trigram extension) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Infrastructure | Docker Compose |

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key (or Ollama for local LLM)

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repository-url>
cd arrow_task

python -m venv myenv
myenv\Scripts\activate    # Windows
# source myenv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
DATABASE_URL=postgresql://movie_agent:movie_agent_pass@localhost:5432/movie_agent

# LLM Provider: "openai" or "ollama"
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# For fully local execution with Ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.1:8b

# Embeddings (local sentence-transformers by default)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

This starts a PostgreSQL 16 container with pgvector and pg_trgm extensions. The schema is automatically initialized from `db/init.sql`.

### 4. Ingest the data

Place the TMDB CSV files in the `data/` directory:
- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

Run the ingestion pipeline:

```bash
python -m movie_agent.data.ingest
```

This loads the CSVs, preprocesses the data, generates embeddings using sentence-transformers, and bulk-inserts everything into PostgreSQL.

### 5. Run the application

**Streamlit UI:**

```bash
streamlit run movie_agent/ui/app.py
```

**FastAPI server:**

```bash
uvicorn movie_agent.api:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ask` | Ask a question (returns full answer) |
| POST | `/ask/stream` | Ask with streaming (Server-Sent Events) |

### Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 5 highest rated sci-fi movies?"}'
```

## Project Structure

```
arrow_task/
├── docker-compose.yml          # PostgreSQL + pgvector service
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── ARCHITECTURE.md             # Detailed architecture documentation
│
├── db/
│   └── init.sql                # Database schema, indexes, extensions
│
├── data/                       # TMDB CSV files
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── models/
│   └── all-MiniLM-L6-v2/      # Local embedding model cache
│
└── movie_agent/                # Main application package
    ├── config.py               # Centralized settings
    ├── api.py                  # FastAPI REST API
    │
    ├── agent/                  # Agent layer (LangGraph)
    │   ├── router.py           # Router agent (delegates to sub-agents)
    │   ├── sql_agent.py        # SQL query agent
    │   ├── semantic_search_agent.py  # Vector similarity agent
    │   ├── fuzzy_search_agent.py     # Title matching agent
    │   ├── tools.py            # Tool definitions
    │   └── prompts.py          # System prompts for all agents
    │
    ├── data/                   # Data processing pipeline
    │   ├── loader.py           # CSV loading
    │   ├── preprocessing.py    # JSON parsing, normalization
    │   ├── ingest.py           # Full ingestion pipeline
    │   └── models.py           # Pydantic models
    │
    ├── db/                     # Database access layer
    │   ├── connection.py       # Connection pool
    │   └── queries.py          # SQL query execution
    │
    ├── rag/                    # RAG pipeline
    │   ├── embeddings.py       # Sentence-transformers model
    │   ├── retriever.py        # Vector retrieval from Postgres
    │   └── document_builder.py # Movie document construction
    │
    ├── search/                 # Search implementations
    │   ├── fuzzy.py            # pg_trgm trigram similarity
    │   ├── semantic.py         # pgvector cosine similarity
    │   ├── structured.py       # SQL-based filtering
    │   └── hybrid.py           # Combined SQL + vector search
    │
    └── ui/                     # Streamlit UI
        ├── app.py              # Chat interface
        └── components.py       # Reusable widgets
```

## Architecture

The system follows a data-first approach with PostgreSQL as the single source of truth:

```
┌─────────────────────────────────────────────────┐
│            Streamlit UI / FastAPI                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Router Agent (LangGraph)            │
│    Decides: SQL | Semantic | Fuzzy search        │
└───┬────────────────┬───────────────────┬────────┘
    │                │                   │
    ▼                ▼                   ▼
┌────────┐   ┌────────────┐   ┌──────────────┐
│SQL Agent│   │Semantic    │   │Fuzzy Search  │
│(queries)│   │Search Agent│   │Agent (titles)│
└───┬─────┘   └─────┬──────┘   └──────┬───────┘
    │                │                  │
    ▼                ▼                  ▼
┌─────────────────────────────────────────────────┐
│       PostgreSQL + pgvector + pg_trgm           │
│  (movies, genres, embeddings, trigram indexes)  │
└─────────────────────────────────────────────────┘
```

## Example Queries

| Query Type | Example |
|------------|---------|
| Structured | "How many action movies were released after 2010?" |
| Structured | "Top 5 highest budget movies with rating above 7" |
| Semantic | "A movie about time travel and love" |
| Semantic | "Something dark and mysterious set in space" |
| Fuzzy title | "Tell me about Interstellar" |
| Fuzzy title | "What is Incpetion about?" |

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://movie_agent:movie_agent_pass@localhost:5432/movie_agent` | PostgreSQL connection string |
| `LLM_PROVIDER` | `openai` | LLM provider (`openai` or `ollama`) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `FUZZY_THRESHOLD` | `70` | Minimum similarity score for fuzzy matching (0-100) |
| `SEMANTIC_TOP_K` | `10` | Number of results for semantic search |

## Data Source

This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle containing ~5000 movies with metadata including budget, revenue, ratings, genres, cast, crew, and plot overviews.

## License

This project is for educational and research purposes.
