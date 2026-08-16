"""Data ingestion pipeline: CSV -> preprocess -> embed -> PostgreSQL.

Run with: python ./movie_agent/data/ingest.py

NOTE: Tables are created automatically by db/init.sql when the PostgreSQL
container starts. This script only handles data insertion.
"""

import sys
from pathlib import Path

# Add project root to sys.path so package imports work when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import psycopg2
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from movie_agent.config import DATABASE_URL
from movie_agent.data.loader import load_movies
from movie_agent.data.preprocessing import preprocess, extract_json_tables
from movie_agent.rag.embeddings import encode_texts


# ============================================================
# SQL: Insert statements
# ============================================================
INSERT_MOVIES_SQL = """
    INSERT INTO movies (id, budget, homepage, original_language, original_title,
                        overview, popularity, revenue, runtime,
                        status, tagline, title, vote_average, vote_count,
                        embedding)
    VALUES %s
    ON CONFLICT (id) DO NOTHING
"""


def _safe_int(value):
    """Convert value to int, returning None for NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def _safe_float(value):
    """Convert value to float, returning None for NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def prepare_movie_rows(df, embeddings) -> list[tuple]:
    """Convert DataFrame rows + embeddings into tuples for insertion."""
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append((
            int(row["id"]),
            _safe_int(row["budget"]),
            row["homepage"] if pd.notna(row["homepage"]) else None,
            row["original_language"],
            row["original_title"],
            row["overview"] if pd.notna(row["overview"]) else None,
            _safe_float(row["popularity"]),
            _safe_int(row["revenue"]),
            _safe_float(row["runtime"]),
            row["status"],
            row["tagline"] if pd.notna(row["tagline"]) else None,
            row["title"],
            _safe_float(row["vote_average"]),
            _safe_int(row["vote_count"]),
            embeddings[i].tolist(),
        ))
    return rows


def insert_movies(conn, rows):
    """Bulk insert rows into movies."""
    with conn.cursor() as cur:
        execute_values(cur, INSERT_MOVIES_SQL, rows)
    conn.commit()
    print(f"Inserted {len(rows)} rows into movies.")


def insert_lookup_and_junction(conn, json_tables: dict):
    """Insert lookup tables and junction tables for the 4 JSON columns."""

    with conn.cursor() as cur:
        # --- GENRES ---
        genres_lookup = json_tables["genres"]["lookup"]
        genres_rows = [
            (int(row["id"]), row["name"])
            for _, row in genres_lookup.iterrows()
        ]
        if genres_rows:
            execute_values(cur, """
                INSERT INTO genres (id, name) VALUES %s ON CONFLICT (id) DO NOTHING
            """, genres_rows)

        genres_exploded = json_tables["genres"]["exploded"]
        genre_junction = []
        for movie_id, row in genres_exploded.iterrows():
            if "id" in row and row["id"] is not None and not (isinstance(row["id"], float) and row["id"] != row["id"]):
                genre_junction.append((int(movie_id), int(row["id"])))
        if genre_junction:
            execute_values(cur, """
                INSERT INTO movie_genres (movie_id, genre_id) VALUES %s ON CONFLICT DO NOTHING
            """, genre_junction)

        # --- PRODUCTION COMPANIES ---
        companies_lookup = json_tables["production_companies"]["lookup"]
        companies_rows = [
            (int(row["id"]), row["name"])
            for _, row in companies_lookup.iterrows()
        ]
        if companies_rows:
            execute_values(cur, """
                INSERT INTO production_companies (id, name) VALUES %s ON CONFLICT (id) DO NOTHING
            """, companies_rows)

        companies_exploded = json_tables["production_companies"]["exploded"]
        company_junction = []
        for movie_id, row in companies_exploded.iterrows():
            if "id" in row and row["id"] is not None and not (isinstance(row["id"], float) and row["id"] != row["id"]):
                company_junction.append((int(movie_id), int(row["id"])))
        if company_junction:
            execute_values(cur, """
                INSERT INTO movie_companies (movie_id, company_id) VALUES %s ON CONFLICT DO NOTHING
            """, company_junction)

        # --- PRODUCTION COUNTRIES ---
        countries_lookup = json_tables["production_countries"]["lookup"]
        countries_rows = [
            (row["iso_3166_1"], row["name"])
            for _, row in countries_lookup.iterrows()
        ]
        if countries_rows:
            execute_values(cur, """
                INSERT INTO production_countries (iso_3166_1, name) VALUES %s ON CONFLICT (iso_3166_1) DO NOTHING
            """, countries_rows)

        countries_exploded = json_tables["production_countries"]["exploded"]
        country_junction = []
        for movie_id, row in countries_exploded.iterrows():
            if "iso_3166_1" in row and row["iso_3166_1"] is not None and isinstance(row["iso_3166_1"], str):
                country_junction.append((int(movie_id), row["iso_3166_1"]))
        if country_junction:
            execute_values(cur, """
                INSERT INTO movie_countries (movie_id, country_code) VALUES %s ON CONFLICT DO NOTHING
            """, country_junction)

        # --- SPOKEN LANGUAGES ---
        languages_lookup = json_tables["spoken_languages"]["lookup"]
        languages_rows = [
            (row["iso_639_1"], row["name"])
            for _, row in languages_lookup.iterrows()
        ]
        if languages_rows:
            execute_values(cur, """
                INSERT INTO spoken_languages (iso_639_1, name) VALUES %s ON CONFLICT (iso_639_1) DO NOTHING
            """, languages_rows)

        languages_exploded = json_tables["spoken_languages"]["exploded"]
        language_junction = []
        for movie_id, row in languages_exploded.iterrows():
            if "iso_639_1" in row and row["iso_639_1"] is not None and isinstance(row["iso_639_1"], str):
                language_junction.append((int(movie_id), row["iso_639_1"]))
        if language_junction:
            execute_values(cur, """
                INSERT INTO movie_languages (movie_id, language_code) VALUES %s ON CONFLICT DO NOTHING
            """, language_junction)

    conn.commit()
    print("Lookup and junction tables populated.")


def run():
    """Execute the full ingestion pipeline."""
    # 1. Load data
    print("=" * 50)
    print("STEP 1: Loading CSV data")
    print("=" * 50)
    movies_df = load_movies()

    # 2. Preprocess flat columns
    print("\n" + "=" * 50)
    print("STEP 2: Preprocessing")
    print("=" * 50)
    df = preprocess(movies_df)

    # 3. Extract JSON columns into lookup/junction tables
    print("\n" + "=" * 50)
    print("STEP 3: Extracting JSON columns")
    print("=" * 50)
    json_tables = extract_json_tables(movies_df)

    # 4. Generate embeddings
    print("\n" + "=" * 50)
    print("STEP 4: Generating embeddings")
    print("=" * 50)
    overviews = df["overview"].fillna("").tolist()
    embeddings = encode_texts(overviews)

    # 5. Create tables and insert
    print("\n" + "=" * 50)
    print("STEP 5: Inserting into PostgreSQL")
    print("=" * 50)
    conn = psycopg2.connect(DATABASE_URL)
    try:
        # Insert movies
        rows = prepare_movie_rows(df, embeddings)
        insert_movies(conn, rows)

        # Insert lookup + junction tables
        insert_lookup_and_junction(conn, json_tables)

        # Verify
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM movies;")
            print(f"\n  movies: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM genres;")
            print(f"  genres: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM production_companies;")
            print(f"  production_companies: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM production_countries;")
            print(f"  production_countries: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM spoken_languages;")
            print(f"  spoken_languages: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM movie_genres;")
            print(f"  movie_genres: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM movie_companies;")
            print(f"  movie_companies: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM movie_countries;")
            print(f"  movie_countries: {cur.fetchone()[0]} rows")
            cur.execute("SELECT COUNT(*) FROM movie_languages;")
            print(f"  movie_languages: {cur.fetchone()[0]} rows")
    finally:
        conn.close()

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    run()
