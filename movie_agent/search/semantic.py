"""Semantic search: embed query with sentence-transformers, then retrieve
top-k similar movies from PostgreSQL using pgvector cosine similarity."""

import psycopg2
from movie_agent.config import DATABASE_URL, SEMANTIC_TOP_K
from movie_agent.rag.embeddings import encode_single


def semantic_search(query: str, top_k: int | None = None) -> list[dict]:
    """Search movies by semantic similarity to the query.

    Embeds the query using sentence-transformers (all-MiniLM-L6-v2),
    then retrieves the top-k most similar movies using cosine distance
    against the precomputed embeddings stored in PostgreSQL (pgvector).

    Args:
        query: Natural language search query (e.g. "space adventure with aliens").
        top_k: Number of results to return. Defaults to SEMANTIC_TOP_K from config.

    Returns:
        List of dicts with movie info and similarity score, ordered by relevance.
    """
    if top_k is None:
        top_k = SEMANTIC_TOP_K

    # Embed the query
    query_embedding = encode_single(query)

    # Query PostgreSQL with pgvector cosine similarity
    # The operator <=> is cosine distance (1 - cosine_similarity)
    sql = """
        SELECT
            id,
            title,
            overview,
            release_date,
            vote_average,
            popularity,
            1 - (embedding <=> %s::vector) AS similarity
        FROM movies
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            embedding_str = str(query_embedding)
            cur.execute(sql, (embedding_str, embedding_str, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "title": row[1],
            "overview": row[2],
            "release_date": str(row[3]) if row[3] else None,
            "vote_average": row[4],
            "popularity": row[5],
            "similarity": round(row[6], 4),
        })

    return results
