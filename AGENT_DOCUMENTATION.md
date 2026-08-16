# Agent Documentation — Technical Design Document

## TMDB Movie Agentic AI System

---

## 1. Data Decisions

### Dataset Files Used

The system ingests two CSV files from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle):

| File | Records | Description |
|------|---------|-------------|
| `tmdb_5000_movies.csv` | ~4,803 | Movie metadata: budget, revenue, genres, overview, keywords, production info |
| `tmdb_5000_credits.csv` | ~4,803 | Cast and crew information per movie |

### Number of Movies Processed

Approximately **4,803 movies** are processed through the ingestion pipeline. After deduplication and removal of entries with missing IDs, the final count in PostgreSQL is typically 4,800–4,803 rows in the `movies` table.

### Join Strategy

The current implementation loads `tmdb_5000_movies.csv` as the primary table. The credits file is available for loading via `loader.py` (`load_credits()` and `load_all()`), but the core ingestion pipeline (`ingest.py`) processes the movies file directly. The join would occur on `movies.id = credits.movie_id`.

This design decision was made because:
- The movies CSV already contains sufficient metadata for all three search modes
- Credit data (cast/crew) is valuable for enhanced queries but adds pipeline complexity
- The system can be extended to incorporate credits without schema changes

### JSON Fields Parsed

Four columns in the movies CSV contain JSON arrays that are parsed and normalized into separate lookup and junction tables:

| JSON Column | Keys Extracted | Lookup Table | Junction Table |
|-------------|---------------|--------------|----------------|
| `genres` | `id`, `name` | `genres` | `movie_genres` |
| `production_companies` | `id`, `name` | `production_companies` | `movie_companies` |
| `production_countries` | `iso_3166_1`, `name` | `production_countries` | `movie_countries` |
| `spoken_languages` | `iso_639_1`, `name` | `spoken_languages` | `movie_languages` |

The parsing approach (`explode_json_column()` in `preprocessing.py`):
1. Parse JSON string into Python list of dicts
2. Explode into one row per item (indexed by movie_id)
3. Extract only the relevant keys
4. Create a deduplicated lookup table
5. Retain the exploded table as the junction mapping

### Missing-Value Handling

The preprocessing pipeline applies the following rules:

| Field | Condition | Action | Rationale |
|-------|-----------|--------|-----------|
| `budget` | `== 0` | Set to `NULL` | Zero is a sentinel for "unknown," not a real $0 budget |
| `revenue` | `== 0` | Set to `NULL` | Same reasoning as budget |
| `runtime` | `== 0` | Set to `NULL` | A 0-minute movie is clearly missing data |
| `overview` | `NaN` | Stored as `NULL`, treated as empty string for embedding | Avoids embedding errors while preserving DB semantics |
| `homepage` | `NaN` | Set to `NULL` | Optional field |
| `tagline` | `NaN` | Set to `NULL` | Optional field |
| `release_date` | String | Converted to `DATE` type | Enables date-range filtering |
| All columns | `NaN` | Replaced with `None` via `df.where(pd.notna(df), None)` | PostgreSQL-compatible NULL |

This approach ensures the SQL Agent never produces misleading aggregations (e.g., `AVG(budget)` is not dragged down by thousands of fake zeros).

### Fields Used for Structured Search (SQL Agent)

The SQL Agent operates over the full relational schema. Key queryable fields:

- **Numeric filtering**: `vote_average`, `vote_count`, `budget`, `revenue`, `runtime`, `popularity`
- **Date filtering**: `release_date` (supports year extraction, range queries)
- **Text filtering**: `title`, `original_title`, `original_language`, `status`
- **Relational filtering** (via JOINs): genres, production companies, production countries, spoken languages
- **Aggregation targets**: COUNT, AVG, SUM, MIN, MAX over any numeric column
- **Sorting**: Any column, ASC/DESC

### Fields Used for Fuzzy Search

- **Primary field**: `title` (with GIN trigram index via `pg_trgm`)
- **Secondary field**: `original_title` (returned in results for disambiguation)
- **Index**: `CREATE INDEX idx_movies_title_trgm ON movies USING gin (original_title gin_trgm_ops)`

The trigram index enables PostgreSQL's `similarity()` function to operate efficiently over all 4,800+ titles without a sequential scan.

### Fields Used for Semantic Retrieval

- **Embedded field**: `overview` — the movie plot description
- **Embedding column**: `embedding vector(384)` stored directly in the `movies` table
- **Index**: HNSW index with `vector_cosine_ops` (`m=16`, `ef_construction=64`)
- **Returned context fields**: `title`, `overview`, `release_date`, `vote_average`, `popularity`

The decision to embed only `overview` (rather than a composite document) was made because:
1. Overviews contain the richest semantic content for plot/theme/mood matching
2. 384-dimensional vectors from MiniLM-L6-v2 can capture the gist of a 1-2 paragraph overview effectively
3. Adding metadata to the embedded text would dilute semantic signal with structured data better handled by SQL filtering

### What Information is Preserved for Generation

When the agent constructs a final answer, it has access to:
- Full movie records from query results (all columns except `embedding`)
- Similarity scores for semantic matches
- Trigram similarity scores for fuzzy matches
- Relational data through JOINs (genre names, company names, country names, language names)

The LLM receives these as tool outputs and synthesizes a natural-language response grounded in the returned data.

### Discussion of Relevant Fields

| Field | Role in System | Design Consideration |
|-------|---------------|---------------------|
| `genres` | Structured filtering via junction table JOIN | Stored normalized in separate table; supports multi-genre AND/OR queries |
| `overview` | Primary semantic search target; embedded as 384-dim vector | Most semantically rich field; used for plot/theme matching |
| `production_companies` | Structured filtering | Normalized; enables "movies by Studio X" queries |
| `production_countries` | Structured filtering | ISO 3166-1 codes enable consistent geographic queries |
| `spoken_languages` | Structured filtering | ISO 639-1 codes; enables "movies in French" queries |
| `vote_average` | Structured filtering + sorting + aggregation | REAL type; core ranking metric |
| `vote_count` | Confidence weighting in queries | Avoids recommending obscure films with a single 10/10 vote |
| `budget` / `revenue` | Structured filtering + aggregation | NULLable (zeros treated as unknown); enables "highest grossing" queries |
| `runtime` | Structured filtering | NULLable; enables "short films" or "movies under 2 hours" queries |
| `release_date` | Date range filtering, year extraction | DATE type; index on `release_date` for range scans |
| `keywords` | Available in raw CSV but not currently in schema | Future enhancement: could enrich embeddings or enable keyword-based filtering |
| `cast` / `crew` | Available via credits CSV | Currently not ingested; would enable "movies with Actor X" or "directed by Y" queries |

---

## 2. Agent Design

### How Routing Works

The system uses a **hierarchical multi-agent architecture** with a single Router Agent at the top:

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   Router Agent (LangGraph)  │
│   Model: GPT-4o-mini        │
│   Pattern: ReAct             │
│                             │
│   Decides which tool to call │
│   based on query analysis    │
└──────┬──────┬──────┬────────┘
       │      │      │
       ▼      ▼      ▼
   sql_agent  semantic  fuzzy_title
   _tool      _search   _search
              _tool     _tool
```

The Router Agent receives the user's query and its system prompt contains explicit guidelines for delegation:

1. **Factual/structured questions** → `sql_agent_tool`
2. **Descriptive/thematic questions** → `semantic_search_tool`
3. **Specific movie title mentions** → `fuzzy_title_search_tool`
4. **Mixed queries** → Multiple tools in sequence

The Router is a LangGraph ReAct agent — it can reason about which tool to call, observe the result, and decide whether to call additional tools or produce a final answer.

### Which Tools Exist

| Tool | Wraps | Input | Output |
|------|-------|-------|--------|
| `sql_agent_tool` | SQL Agent (its own ReAct agent) | `question: str` — natural language question | Answer text based on SQL query results |
| `semantic_search_tool` | Semantic Search Agent | `query: str` — description/theme | Answer text with similar movies |
| `fuzzy_title_search_tool` | Fuzzy Search Agent | `title: str` — movie title (possibly misspelled) | Answer text with matched movies |

Each tool wraps a complete sub-agent that has its own tools:

**SQL Agent tools** (from LangChain):
- `QuerySQLDatabaseTool` — Execute SQL queries
- `InfoSQLDatabaseTool` — Get table schemas/info
- `ListSQLDatabaseTool` — List available tables

**Semantic Search Agent tool:**
- `search_movies_by_description` — Embeds query, retrieves top-k by cosine similarity

**Fuzzy Search Agent tool:**
- `search_movie_by_title` — Executes `pg_trgm` similarity search on titles

### Boundaries of Each Tool

| Tool | Handles | Does NOT Handle |
|------|---------|-----------------|
| SQL Agent | Counts, rankings, averages, exact filtering by year/rating/budget/genre, comparisons, GROUP BY | Semantic meaning, plot descriptions, mood-based queries |
| Semantic Search | Plot/theme/mood matching, "movies like X", descriptive queries | Exact counts, precise filtering, aggregations |
| Fuzzy Search | Title lookup (even misspelled), "what is X about?" | Complex queries, recommendations, data analysis |

### When Multiple Tools Are Invoked

The Router Agent can invoke multiple tools in a single turn when the query spans boundaries:

**Example: "Recommend a highly-rated sci-fi movie about AI"**
- This has both a structured component (highly-rated, sci-fi genre) and a semantic component (about AI)
- The Router may call `sql_agent_tool` for "highly-rated sci-fi movies" AND `semantic_search_tool` for "movie about artificial intelligence"
- It then synthesizes both result sets into a unified answer

**Example: "Tell me about Inception and find similar movies"**
- `fuzzy_title_search_tool` for "Inception" → gets the movie details
- `semantic_search_tool` with Inception's description → finds thematically similar movies

### Why This Architecture Was Selected

**Decision: Hierarchical multi-agent over single-agent-with-many-tools**

Rationale:
1. **Separation of concerns** — Each sub-agent has a focused system prompt optimized for its domain. The SQL Agent knows the schema; the Semantic Agent knows how to interpret similarity scores.
2. **Error isolation** — If the SQL Agent encounters a bad query, it can retry internally without affecting the Router's context.
3. **Scalability** — New capabilities (e.g., a recommendation agent, a comparison agent) can be added as new tools without modifying existing sub-agents.
4. **Prompt efficiency** — Each sub-agent carries only its own system prompt, avoiding a single massive prompt that degrades LLM performance.

**Decision: LangGraph ReAct over LangChain AgentExecutor**

Rationale:
1. LangGraph provides checkpointing for conversation memory (`InMemorySaver`)
2. Stream support at the graph level enables token-by-token UI updates
3. More control over execution flow compared to the legacy AgentExecutor
4. Native support for tool selection reasoning → observation → decision cycles

**Decision: GPT-4o-mini over larger models**

Rationale:
1. Cost efficiency — routing and SQL generation don't require GPT-4o's full reasoning
2. Speed — faster inference means better streaming UX
3. The structured nature of tool use (schema-aware, prompt-guided) compensates for model size

### Worked Examples

#### Example 1: Structured Query with Entity Resolution

**Query:** "How many movies have Christopher Nolan as director?"

```
Router Agent:
  Thought: This is a factual/count question about a specific person → sql_agent_tool
  Action: sql_agent_tool("How many movies have Christopher Nolan as director?")

SQL Agent:
  Thought: I need to check what columns are available for director info
  Action: InfoSQLDatabaseTool → sees no 'director' column in movies
  Thought: The credits/crew data may not be in the current schema. Let me try
           searching by other means or checking if director is stored.
  Action: QuerySQLDatabaseTool →
    SELECT COUNT(*) FROM movies
    WHERE overview ILIKE '%christopher nolan%'
       OR tagline ILIKE '%nolan%';
  Observation: Limited results since director isn't a direct column

Router Agent:
  Final Answer: Based on available data, [provides answer with caveat about
  director field availability]
```

**Note:** This query exposes a current limitation — the credits/crew data (containing director information) is not yet ingested into the database. The SQL Agent must fall back to text search or acknowledge the limitation.

#### Example 2: Semantic + Structured (Hybrid)

**Query:** "Find me a dark science-fiction movie about artificial intelligence"

```
Router Agent:
  Thought: This has a descriptive/thematic component ("dark... about AI")
           → semantic_search_tool
  Action: semantic_search_tool("dark science-fiction movie about artificial intelligence")

Semantic Search Agent:
  Action: search_movies_by_description(
    query="dark science-fiction movie about artificial intelligence",
    top_k=5
  )
  Observation: Returns movies like Ex Machina, Blade Runner, A.I., etc.
  with similarity scores

Router Agent:
  Final Answer: Presents top semantic matches with ratings and overviews,
  highlighting why each fits the "dark sci-fi AI" criteria
```

---

## 3. Search System

### Structured Search (SQL Agent)

The SQL Agent translates natural language into SQL queries against the normalized PostgreSQL schema.

#### Filtering

The agent generates WHERE clauses for:
- **Numeric conditions**: `vote_average > 7.5`, `budget > 100000000`, `runtime < 120`
- **Date ranges**: `release_date >= '2010-01-01'`, `EXTRACT(YEAR FROM release_date) = 2015`
- **Text matching**: `original_language = 'en'`, `status = 'Released'`
- **Relational filters** (via JOIN):
  ```sql
  SELECT m.title, m.vote_average
  FROM movies m
  JOIN movie_genres mg ON m.id = mg.movie_id
  JOIN genres g ON mg.genre_id = g.id
  WHERE g.name = 'Action'
  ```

#### Sorting

Any column can be used for ORDER BY with ASC/DESC. Common patterns:
- `ORDER BY vote_average DESC` — highest rated
- `ORDER BY popularity DESC` — most popular
- `ORDER BY revenue DESC NULLS LAST` — highest grossing (NULL-safe)
- `ORDER BY release_date DESC` — newest

#### Aggregation

The SQL Agent can produce:
- `COUNT(*)` — number of movies matching criteria
- `AVG(vote_average)` — average rating of a subset
- `SUM(revenue)` — total revenue
- `GROUP BY` with genre, year, language for distribution analysis

#### Date Ranges

```sql
-- Movies from a specific decade
WHERE release_date >= '2000-01-01' AND release_date < '2010-01-01'

-- Movies from a specific year
WHERE EXTRACT(YEAR FROM release_date) = 2014
```

#### Multi-Condition Queries

The agent composes complex AND/OR conditions:
```sql
SELECT title, vote_average, runtime
FROM movies m
JOIN movie_genres mg ON m.id = mg.movie_id
JOIN genres g ON mg.genre_id = g.id
WHERE g.name = 'Science Fiction'
  AND vote_average >= 7.0
  AND runtime BETWEEN 90 AND 150
  AND release_date >= '2000-01-01'
ORDER BY vote_average DESC
LIMIT 10;
```

#### Safety Constraints

The SQL Agent system prompt enforces:
1. Never SELECT the `embedding` column (binary vector data)
2. Default LIMIT 10 unless user asks for more
3. Retry on query failure with error inspection
4. Handle NULLs in budget/revenue/runtime aggregations

### Fuzzy Search

The fuzzy search system uses PostgreSQL's `pg_trgm` extension for trigram-based similarity matching directly in the database.

#### How Trigram Similarity Works

PostgreSQL's `pg_trgm` breaks strings into sets of 3-character subsequences (trigrams) and measures the overlap between two sets:

```
"Avatar" → {" a", " av", "ava", "vat", "ata", "tar", "ar "}
"Avater" → {" a", " av", "ava", "vat", "ate", "ter", "er "}
similarity = |intersection| / |union| ≈ 0.5+
```

#### Title Normalization

The database handles normalization internally — the `similarity()` function operates on the stored `title` column values. The GIN trigram index (`idx_movies_title_trgm`) accelerates the similarity computation.

#### Threshold Configuration

- Default threshold: `FUZZY_THRESHOLD = 70` (in config) → `0.7` similarity score
- This means a title must share at least 70% of its trigrams with the query to be returned
- The threshold can be adjusted lower for more permissive matching (catching worse typos) or higher for stricter matching

The threshold is set relatively low (0.7) because:
- Movie titles can be short (2-3 words), where a single typo represents a significant trigram difference
- Users often provide partial titles ("dark knight" instead of "The Dark Knight Rises")
- It's better to return a slightly larger result set and let the agent pick the best match

#### Ambiguity Handling

When multiple movies have similar scores:
1. Results are returned ordered by `sim_score DESC`
2. The Fuzzy Search Agent presents top matches with their similarity scores
3. The agent can suggest alternatives if the top match isn't confident
4. If no results exceed the threshold, the agent recommends trying a different spelling

#### Query Structure

```sql
SELECT id, title, original_title, overview, release_date,
       vote_average, popularity,
       similarity(title, %s) AS sim_score
FROM movies
WHERE similarity(title, %s) > %s
ORDER BY sim_score DESC
LIMIT %s;
```

---

## 4. RAG Pipeline

### Document Construction

The embedding input is the movie's `overview` field — the plot description/synopsis stored in the TMDB dataset. For movies with missing overviews, an empty string is used (producing a near-zero vector that won't match strongly against any query).

The decision to use raw `overview` rather than a composite document (combining title, genres, cast, etc.) was deliberate:
- **Semantic purity**: The embedding captures plot/theme semantics without noise from structured metadata
- **Separation of concerns**: Structured attributes are better handled by SQL filtering
- **Embedding model capacity**: MiniLM-L6-v2 has a 256-token limit; overviews fit naturally, but composite documents might be truncated

### Embedding Model

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Max sequence length | 256 tokens |
| Architecture | BERT-based, 6 transformer layers |
| Normalization | L2-normalized output (2_Normalize layer) |
| Pooling | Mean pooling (1_Pooling layer) |
| Size | ~80MB |
| Execution | Local CPU (no API calls needed) |

**Why this model:**
1. **Free and local** — No per-embedding API costs; runs entirely on CPU
2. **Fast** — Encodes all 4,800 movies in 2-5 minutes on typical hardware
3. **384 dimensions** — Small enough for efficient storage/indexing, large enough for good semantic discrimination
4. **Well-benchmarked** — Strong performance on semantic textual similarity tasks
5. **Cached locally** — Model files saved to `models/all-MiniLM-L6-v2/` for offline use

### Vector Database / Index

Vectors are stored directly in PostgreSQL using the `pgvector` extension:

- **Storage**: `embedding vector(384)` column in the `movies` table
- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Distance metric**: Cosine distance (`vector_cosine_ops`)
- **Index parameters**: `m = 16`, `ef_construction = 64`

```sql
CREATE INDEX idx_movies_embedding ON movies
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**Why pgvector over a dedicated vector DB (Pinecone, Weaviate, Qdrant):**
1. **Unified store** — One database for structured data + vectors + memory (no synchronization issues)
2. **ACID transactions** — Embeddings are always consistent with their associated movie data
3. **Combined queries** — Can apply SQL WHERE filters + vector ORDER BY in a single query
4. **Operational simplicity** — One container to manage, one connection pool
5. **Scale is manageable** — 4,800 vectors × 384 dims fits comfortably; HNSW handles this efficiently

### Retrieval Strategy

The retrieval uses cosine similarity via pgvector's `<=>` operator:

```sql
SELECT id, title, overview, release_date, vote_average, popularity,
       1 - (embedding <=> %s::vector) AS similarity
FROM movies
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s;
```

The `<=>` operator computes cosine distance (1 - cosine_similarity). We order by ascending distance (closest first) and compute similarity as `1 - distance` for the returned score.

### Top-k Configuration

- **Default**: `SEMANTIC_TOP_K = 10` (configurable via `.env`)
- **Agent-controlled**: The `search_movies_by_description` tool accepts a `top_k` parameter (capped at 20)
- **Rationale**: 10 provides enough variety for recommendations while keeping context manageable for the LLM

### Metadata Filtering

The retriever (`rag/retriever.py`) supports optional pre-filtering before vector search:

```python
def retrieve_with_filter(query, top_k=None, min_vote_average=None, min_year=None, max_year=None):
```

This adds WHERE clauses before the ORDER BY vector distance:

```sql
SELECT ... FROM movies
WHERE embedding IS NOT NULL
  AND vote_average >= %s          -- optional
  AND EXTRACT(YEAR FROM release_date) >= %s  -- optional
  AND EXTRACT(YEAR FROM release_date) <= %s  -- optional
ORDER BY embedding <=> %s::vector
LIMIT %s;
```

This is a **pre-filtering** approach (filter first, then rank by similarity) rather than post-filtering. The tradeoff:
- **Pro**: Guarantees all results meet the filter criteria
- **Con**: May miss semantically excellent matches that barely fail the filter

### Reranking

No explicit reranking step is currently implemented. Results are returned in raw cosine similarity order. Potential future additions:
- Cross-encoder reranking for the top-k
- Score normalization across different query types
- Popularity-weighted scoring

### Context Construction

The semantic search agent receives tool output as a formatted string:

```
1. **Ex Machina** (similarity: 0.7842)
   Rating: 7.6 | Released: 2015-01-21
   Caleb, a 26-year-old programmer at the world's largest internet company...

2. **Blade Runner 2049** (similarity: 0.7531)
   Rating: 7.3 | Released: 2017-10-04
   Thirty years after the events of the first film...
```

This becomes the agent's observation, which it uses to craft a final answer.

### Generation Model

- **Model**: GPT-4o-mini (via OpenAI-compatible API)
- **Temperature**: 0 (deterministic output)
- **Role**: The LLM synthesizes search results into conversational answers

### Hallucination Mitigation

1. **Grounded prompts**: Each agent's system prompt instructs it to answer only based on tool output
2. **Explicit tool results**: The LLM sees the actual data (titles, scores, overviews) — it doesn't need to recall movie facts from training data
3. **Temperature 0**: Eliminates sampling randomness
4. **Schema awareness**: The SQL Agent prompt includes the full schema, reducing hallucinated column names
5. **Retry on failure**: If a SQL query fails, the agent sees the error and rewrites (rather than making up results)

---

## 5. Multi-Turn Memory

### Memory Architecture

The system uses LangGraph's `InMemorySaver` checkpointer, keyed by `session_id` (a UUID thread identifier):

```python
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": session_id}}
result = agent.invoke({"messages": [("human", question)]}, config=config)
```

### What State Is Persisted Between Turns

| State | Persisted? | Mechanism |
|-------|-----------|-----------|
| Full message history (user + assistant + tool) | Yes | LangGraph checkpoint stores all messages per thread_id |
| Previous tool calls and results | Yes | Part of message history |
| Conversational context ("the first one", "tell me more") | Yes | LLM sees full history and can resolve references |
| Session identity | Yes | UUID session_id passed across API calls |

### What State Is NOT Persisted

| State | Persisted? | Reason |
|-------|-----------|--------|
| Explicit "last_results" array | No | Not implemented (designed in schema but not wired) |
| Active filters from previous turn | No | Each query is independently routed; no filter inheritance |
| Selected movie context | No | Handled implicitly via message history, not a structured state object |
| Cross-session memory | No | Each session_id is independent; no user-level long-term memory |
| Memory across restarts | No | InMemorySaver is in-process; state is lost when the application restarts |

### Conversational Reference Resolution

The LLM handles references to previous context through its message history:

**Turn 1:** "What are the top 5 action movies?"
→ Agent returns: Movie A, Movie B, Movie C, Movie D, Movie E

**Turn 2:** "Tell me more about the second one"
→ The full Turn 1 exchange (including tool results) is in the message history
→ The LLM can resolve "the second one" = Movie B
→ Routes to fuzzy_title_search_tool("Movie B") or uses existing context

### Session Management (UI Layer)

The Streamlit UI maintains its own session state:

```python
st.session_state.sessions = {
    session_id: {
        "name": "Chat 1",        # Auto-named from first message
        "messages": [...]         # Full UI message history
    }
}
```

This is separate from the LangGraph checkpointer — it handles UI display while LangGraph handles agent memory.

### Database-Level Memory (Designed but Not Active)

The `init.sql` schema includes commented-out tables for persistent memory:

```sql
-- conversations: id, session_id, created_at, updated_at
-- messages: id, conversation_id, role, content, tool_name, tool_input, tool_output
-- session_state: id, session_id, key, value (JSONB)
```

These were designed for:
- **Long-term memory**: Survive app restarts, enable conversation resumption
- **Short-term state**: Track `last_results`, `active_filters`, `current_movie` as JSONB

The current implementation uses `InMemorySaver` for simplicity during development, with the database-backed approach ready for production migration.

---

## 6. Example Queries & Results

### Query 1: Aggregation

**Query:** "How many movies have a rating above 8?"

**Agent path:** Router → `sql_agent_tool`

**Expected SQL:**
```sql
SELECT COUNT(*) FROM movies WHERE vote_average > 8;
```

**Expected output:**
> There are 199 movies in the database with a rating above 8.0.

---

### Query 2: Filtering + Sorting

**Query:** "Show me the top 5 highest-grossing movies released after 2010"

**Agent path:** Router → `sql_agent_tool`

**Expected SQL:**
```sql
SELECT title, revenue, release_date, vote_average
FROM movies
WHERE release_date > '2010-01-01' AND revenue IS NOT NULL
ORDER BY revenue DESC
LIMIT 5;
```

**Expected output:**
> Here are the top 5 highest-grossing movies released after 2010:
> 1. Avatar — $2,787,965,087 (2009-12-10... note: may appear depending on exact date parsing)
> 2. Star Wars: The Force Awakens — $2,068,223,624
> 3. Jurassic World — $1,513,528,810
> 4. Furious 7 — $1,506,249,360
> 5. Avengers: Age of Ultron — $1,405,403,694

---

### Query 3: Multi-Condition with JOIN

**Query:** "What are the best sci-fi movies released between 2005 and 2015 with more than 1000 votes?"

**Agent path:** Router → `sql_agent_tool`

**Expected SQL:**
```sql
SELECT m.title, m.vote_average, m.release_date, m.vote_count
FROM movies m
JOIN movie_genres mg ON m.id = mg.movie_id
JOIN genres g ON mg.genre_id = g.id
WHERE g.name = 'Science Fiction'
  AND m.release_date BETWEEN '2005-01-01' AND '2015-12-31'
  AND m.vote_count > 1000
ORDER BY m.vote_average DESC
LIMIT 10;
```

**Expected output:**
> Here are the best sci-fi movies from 2005-2015 (with 1000+ votes):
> 1. Interstellar (2014) — Rating: 8.1, 8,780 votes
> 2. Inception (2010) — Rating: 8.1, 10,356 votes
> 3. The Martian (2015) — Rating: 7.6, 5,765 votes
> ...

---

### Query 4: Fuzzy Title Lookup

**Query:** "What is Incpetion about?"

**Agent path:** Router → `fuzzy_title_search_tool`

**Fuzzy search SQL:**
```sql
SELECT id, title, original_title, overview, release_date, vote_average, popularity,
       similarity(title, 'Incpetion') AS sim_score
FROM movies
WHERE similarity(title, 'Incpetion') > 0.6
ORDER BY sim_score DESC
LIMIT 5;
```

**Expected output:**
> I found the movie you're looking for! Despite the typo, here's what I found:
>
> **Inception** (match: 0.7143)
> Rating: 8.1 | Released: 2010-07-14
>
> A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.

---

### Query 5: Semantic Search (Theme/Mood)

**Query:** "Find me a movie about a lonely astronaut stranded in space"

**Agent path:** Router → `semantic_search_tool`

**Expected output:**
> Here are movies matching your description of a lonely astronaut stranded in space:
>
> 1. **Gravity** (similarity: 0.72) — Rating: 7.2
>    Dr. Ryan Stone is a medical engineer on her first shuttle mission...
>
> 2. **The Martian** (similarity: 0.68) — Rating: 7.6
>    During a manned mission to Mars, Astronaut Mark Watney is presumed dead...
>
> 3. **Interstellar** (similarity: 0.61) — Rating: 8.1
>    Interstellar chronicles the adventures of a group of explorers...

---

### Query 6: Semantic Search (Mood-Based)

**Query:** "Something dark and psychological, like a thriller about obsession"

**Agent path:** Router → `semantic_search_tool`

**Expected output:**
> Here are some dark psychological thrillers about obsession:
>
> 1. **Black Swan** (similarity: 0.65) — Rating: 7.5
>    A ballet dancer wins the lead in "Swan Lake" and is perfect for the role of the White Swan...
>
> 2. **Shutter Island** (similarity: 0.58) — Rating: 7.8
>    World War II soldier-turned-U.S. Marshal Teddy Daniels investigates the disappearance of a patient...
>
> 3. **Gone Girl** (similarity: 0.56) — Rating: 7.7
>    With his wife's disappearance having become the focus of an intense media circus...

---

### Query 7: Multi-Turn Behavior

**Turn 1 Query:** "What are the top rated horror movies?"

**Agent path:** Router → `sql_agent_tool`

**Output:**
> Here are the top-rated horror movies:
> 1. Psycho (1960) — Rating: 8.2
> 2. The Shining (1980) — Rating: 8.1
> 3. Alien (1979) — Rating: 7.9
> ...

**Turn 2 Query:** "Which of those were made before 1980?"

**Agent path:** Router → `sql_agent_tool` (LLM sees previous context)

**Expected behavior:** The agent, having the previous query and results in its message history, generates a follow-up query filtering the same criteria with an additional date constraint:

```sql
SELECT title, vote_average, release_date
FROM movies m
JOIN movie_genres mg ON m.id = mg.movie_id
JOIN genres g ON mg.genre_id = g.id
WHERE g.name = 'Horror'
  AND release_date < '1980-01-01'
ORDER BY vote_average DESC
LIMIT 10;
```

---

### Query 8: A Query That Does NOT Work Well

**Query:** "Who directed The Dark Knight?"

**Agent path:** Router → `fuzzy_title_search_tool` (finds the movie) or `sql_agent_tool`

**Problem:** The `movies` table does not contain a `director` column. The credits data (`tmdb_5000_credits.csv`) has crew information including directors, but it is not currently ingested into the database.

**Actual behavior:**
- The SQL Agent may attempt to query a non-existent `director` column → error → retry
- It may fall back to `ILIKE` searching the `overview` or `tagline` for the director's name → unreliable
- The Fuzzy Agent can find "The Dark Knight" but only returns overview, rating, and date — not director

**Expected output:**
> I found The Dark Knight (2008) with a rating of 8.2, but I don't have director information in the current database. The available data includes plot overview, ratings, genres, and financial information.

**Why it fails:**
- The credits CSV join is not implemented in the ingestion pipeline
- There is no `crew` or `director` column in the `movies` schema
- This is a known data completeness gap — the system has the raw data (`tmdb_5000_credits.csv`) but doesn't process it

---

## 7. Conclusions

### Known Limitations

1. **Missing credits data** — Director, cast, and crew information from `tmdb_5000_credits.csv` is not ingested. This prevents a significant class of queries ("movies by Christopher Nolan", "films starring Leonardo DiCaprio").

2. **In-memory checkpointer** — Conversation history is lost on application restart. The PostgreSQL-backed memory tables are defined in the schema but not wired into the agent layer.

3. **No hybrid search in practice** — The `hybrid.py` module is a stub. True hybrid queries (SQL filters + vector similarity in one pass) must be done manually through the retriever, but the Router Agent cannot currently compose such queries.

4. **Structured search module is a stub** — `structured.py` is empty; all structured search is handled by the SQL Agent directly via LangChain's SQL tools. This works but limits control over query generation.

### Retrieval Weaknesses

1. **Overview-only embeddings** — Semantic search operates only on the `overview` field. Movies with sparse or generic overviews (e.g., "A young man discovers...") match poorly. A composite document including genres, keywords, and taglines would improve recall.

2. **No cross-field semantic search** — Searching "movies with great cinematography" fails because cinematography quality isn't captured in overviews. The semantic model can only match what's in the embedded text.

3. **Embedding model token limit** — MiniLM-L6-v2 has a 256-token limit. Very long overviews are truncated, losing information from the end of the description.

4. **No query expansion** — A query like "happy movie" won't match overviews containing "joyful" or "uplifting" as well as a model with query expansion would.

### Data-Quality Issues

1. **Zero-value ambiguity** — Budget and revenue zeros are treated as NULL (unknown), but some $0-budget films are real (student films, micro-budget). The system cannot distinguish these.

2. **Inconsistent genres** — Genre classification in TMDB is crowdsourced and sometimes inconsistent (e.g., a "Thriller" that's also tagged "Action" and "Drama").

3. **Outdated dataset** — The TMDB 5000 dataset was released circa 2017. No movies after ~2016 are included. Users asking about recent films will get no results.

4. **Missing overviews** — Some movies have empty overviews, producing near-zero embeddings that never surface in semantic search. These movies are only discoverable via SQL or fuzzy search.

### LLM Limitations

1. **SQL hallucination** — The SQL Agent occasionally generates queries referencing columns that don't exist (e.g., `keywords`, `director`) if the user asks about them. The retry mechanism catches most of these, but not always on the first attempt.

2. **Routing ambiguity** — Queries like "Tell me about a movie with robots" could be either a title lookup ("Robots") or a semantic search. The Router doesn't always choose optimally.

3. **Token cost** — Each sub-agent call is a full LLM invocation. A routed query to the SQL Agent involves at minimum 3 LLM calls (Router + SQL Agent schema check + SQL Agent query generation). For complex queries, this can reach 6-8 calls.

4. **Context window pressure** — Long conversations accumulate message history. Eventually the context window fills and the LLM may miss earlier context.

### Scalability Considerations

1. **Vector index at scale** — HNSW with 4,800 vectors is trivial. At 1M+ vectors, index build time, memory usage, and recall tradeoffs become significant. The current parameters (m=16, ef_construction=64) are conservative.

2. **Connection pooling** — The current implementation uses `SimpleConnectionPool(1, 5)`. Under concurrent load (multiple users), this would need to be increased or replaced with async connection pooling (asyncpg + connection pool).

3. **Embedding generation** — Batch ingestion of 4,800 movies takes 2-5 minutes on CPU. Scaling to millions would require GPU acceleration or an embedding API.

4. **Sub-agent latency** — The hierarchical architecture adds latency (Router LLM call → Sub-agent LLM call → Tool execution). For production, parallel tool execution and response caching would help.

### What I Would Improve With Another Week

1. **Ingest credits data** — Add `director` and `top_cast` columns to the movies table. Enable "movies by X director" and "films starring Y" queries.

2. **Persistent memory** — Wire up the PostgreSQL conversation/session tables. Use `PostgresSaver` from `langgraph-checkpoint-postgres` instead of `InMemorySaver`.

3. **Composite embeddings** — Build a richer document text for embedding: `"{title}. {tagline}. {overview}. Genres: {genres}. Keywords: {keywords}."` This would improve semantic recall significantly.

4. **Implement hybrid search** — Complete the `hybrid.py` module. Allow the Router to request "sci-fi movies about time travel" as a single query combining genre filter + vector similarity.

5. **Add keyword extraction** — Parse the `keywords` JSON column from the movies CSV. Keywords like "time travel", "artificial intelligence", "heist" would dramatically improve structured filtering.

6. **Query caching** — Cache frequently asked queries (both embeddings and SQL results) with a TTL. Most movie data is static.

7. **Evaluation pipeline** — Build a test set of 50+ queries with expected results. Measure retrieval recall@k, routing accuracy, and end-to-end answer quality automatically.

8. **Ollama integration testing** — Verify the full pipeline works with local LLMs (llama3.1:8b). Adjust prompts for smaller model capabilities. This would enable a fully offline, zero-cost deployment.

---

*Document generated for the TMDB Movie Agentic AI System — Technical Design Reference*
