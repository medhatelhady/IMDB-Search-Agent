"""SQL query builders for structured search, vector search, and memory."""

import psycopg2
from movie_agent.config import DATABASE_URL


def execute_query(sql: str, params: tuple = None, fetch: bool = True):
    """Execute a SQL query against the PostgreSQL database.

    Args:
        sql: The SQL query string. Use %s placeholders for parameters.
        params: Optional tuple of parameters to safely inject into the query.
        fetch: If True, return results. If False (for INSERT/UPDATE/DELETE), commit and return rowcount.

    Returns:
        If fetch=True: list of tuples (rows) with column names accessible via .description.
        If fetch=False: number of affected rows.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return {"columns": columns, "rows": rows}
            else:
                conn.commit()
                return cur.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
