"""PostgreSQL connection pool and session factory."""

import psycopg2
from psycopg2 import pool
from movie_agent.config import DATABASE_URL

_connection_pool = None


def get_pool(minconn: int = 1, maxconn: int = 5) -> pool.SimpleConnectionPool:
    """Get or create a connection pool."""
    global _connection_pool
    if _connection_pool is None or _connection_pool.closed:
        _connection_pool = pool.SimpleConnectionPool(minconn, maxconn, DATABASE_URL)
    return _connection_pool


def get_connection():
    """Get a connection from the pool."""
    return get_pool().getconn()


def release_connection(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)


def close_pool():
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool and not _connection_pool.closed:
        _connection_pool.closeall()
        _connection_pool = None
