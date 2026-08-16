"""Fuzzy title matching using PostgreSQL pg_trgm extension.

Uses trigram similarity to find movies whose titles closely match the input,
even with typos or partial names.
"""

import psycopg2
from movie_agent.config import DATABASE_URL, FUZZY_THRESHOLD


def fuzzy_search_title(title: str, top_k: int = 5, threshold: float | None = None) -> list[dict]:
    """Search for movies by fuzzy matching on the title using pg_trgm.

    Args:
        title: The movie title to search for (can contain typos or be partial).
        top_k: Number of results to return.
        threshold: Minimum similarity score (0-1). Defaults to FUZZY_THRESHOLD/100 from config.

    Returns:
        List of dicts with movie info and similarity score, ordered by similarity.
    """
    if threshold is None:
        threshold = FUZZY_THRESHOLD / 100.0  # config stores as int (e.g. 70 -> 0.7)

    sql = """
        SELECT
            id,
            title,
            original_title,
            overview,
            release_date,
            vote_average,
            popularity,
            similarity(title, %s) AS sim_score
        FROM movies
        WHERE similarity(title, %s) > %s
        ORDER BY sim_score DESC
        LIMIT %s;
    """

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (title, title, threshold, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "title": row[1],
            "original_title": row[2],
            "overview": row[3],
            "release_date": str(row[4]) if row[4] else None,
            "vote_average": row[5],
            "popularity": row[6],
            "similarity": round(row[7], 4),
        })

    return results
