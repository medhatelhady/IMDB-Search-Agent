"""Retrieval logic: semantic search using pgvector cosine similarity."""

from movie_agent.rag.embeddings import encode_single
from movie_agent.db.queries import execute_query
from movie_agent.config import SEMANTIC_TOP_K


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """Retrieve the most relevant movies for a natural language query.

    Encodes the query into an embedding, then performs cosine similarity
    search against the movie overview embeddings in PostgreSQL (pgvector).

    Args:
        query: Natural language search query (e.g. "a sci-fi movie about time travel").
        top_k: Number of results to return. Defaults to SEMANTIC_TOP_K from config.

    Returns:
        List of dicts, each containing movie fields and a similarity score.
    """
    top_k = top_k or SEMANTIC_TOP_K

    # 1. Encode the query
    query_embedding = encode_single(query)

    # 2. Search using cosine similarity (pgvector <=> operator)
    sql = """
        SELECT id, title, overview, original_title, release_date,
               vote_average, popularity, runtime, revenue, budget,
               1 - (embedding <=> %s::vector) AS similarity
        FROM movies
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    result = execute_query(sql, (str(query_embedding), str(query_embedding), top_k))

    # 3. Format results as list of dicts
    columns = result["columns"]
    rows = result["rows"]

    movies = []
    for row in rows:
        movie = dict(zip(columns, row))
        movies.append(movie)

    return movies


def retrieve_with_filter(
    query: str,
    top_k: int = None,
    min_vote_average: float = None,
    min_year: int = None,
    max_year: int = None,
) -> list[dict]:
    """Retrieve relevant movies with optional metadata filters.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        min_vote_average: Minimum vote average filter.
        min_year: Minimum release year filter.
        max_year: Maximum release year filter.

    Returns:
        List of dicts with movie fields and similarity score.
    """
    top_k = top_k or SEMANTIC_TOP_K
    query_embedding = encode_single(query)

    # Build dynamic WHERE clause
    conditions = ["embedding IS NOT NULL"]
    params = []

    if min_vote_average is not None:
        conditions.append("vote_average >= %s")
        params.append(min_vote_average)

    if min_year is not None:
        conditions.append("EXTRACT(YEAR FROM release_date) >= %s")
        params.append(min_year)

    if max_year is not None:
        conditions.append("EXTRACT(YEAR FROM release_date) <= %s")
        params.append(max_year)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT id, title, overview, original_title, release_date,
               vote_average, popularity, runtime, revenue, budget,
               1 - (embedding <=> %s::vector) AS similarity
        FROM movies
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    all_params = tuple([str(query_embedding)] + params + [str(query_embedding), top_k])
    result = execute_query(sql, all_params)

    columns = result["columns"]
    rows = result["rows"]

    return [dict(zip(columns, row)) for row in rows]
