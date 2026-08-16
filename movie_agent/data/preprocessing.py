"""Parse JSON fields, normalize columns, handle missing values."""

import json
import numpy as np
import pandas as pd

# Columns that map to the movies table (flat/scalar columns)
SELECTED_COLUMNS = [
    "budget",
    "homepage",
    "id",
    "original_language",
    "original_title",
    "overview",
    "popularity",
    "release_date",
    "revenue",
    "runtime",
    "status",
    "tagline",
    "title",
    "vote_average",
    "vote_count",
]

# JSON columns to explode into lookup + junction tables
JSON_COLUMNS_CONFIG = {
    "genres": {"keys": ["id", "name"], "id_key": "id"},
    "spoken_languages": {"keys": ["iso_639_1", "name"], "id_key": "iso_639_1"},
    "production_companies": {"keys": ["id", "name"], "id_key": "id"},
    "production_countries": {"keys": ["iso_3166_1", "name"], "id_key": "iso_3166_1"},
}


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the columns needed for the movies table."""
    return df[SELECTED_COLUMNS].copy()


def handle_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zeros with NaN in revenue/budget/runtime, then NaN with None for PostgreSQL."""
    df.loc[df.revenue == 0, 'revenue'] = np.nan
    df.loc[df.budget == 0, 'budget'] = np.nan
    df.loc[df.runtime == 0, 'runtime'] = np.nan
    return df.where(pd.notna(df), None)


def handle_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert release_date strings to Python date objects."""
    df['release_date'] = pd.to_datetime(df['release_date']).dt.date
    return df.copy()


def explode_json_column(df: pd.DataFrame, col: str, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a JSON column into an exploded table (movie_id -> item) and a deduplicated lookup table.

    Args:
        df: Raw movies DataFrame (must have 'id' column).
        col: Name of the JSON column to parse.
        keys: List of keys to extract from each dict in the JSON array.

    Returns:
        (exploded_table, lookup_table):
            - exploded_table: DataFrame indexed by movie_id with columns from keys.
            - lookup_table: Deduplicated DataFrame with unique items.
    """
    exploded = (
        df[[col]]
        .set_index(df.id)[col]
        .apply(lambda x: json.loads(x) if isinstance(x, str) else [])
        .explode()
        .apply(pd.Series)
    )

    # Filter to only the keys we care about (some rows may be empty after explode)
    available_keys = [k for k in keys if k in exploded.columns]
    if not available_keys:
        empty = pd.DataFrame(columns=keys)
        return empty, empty

    exploded = exploded[available_keys]
    lookup = exploded.drop_duplicates().dropna().reset_index(drop=True)

    return exploded, lookup


def extract_json_tables(df: pd.DataFrame) -> dict:
    """Extract all JSON columns into lookup and junction tables.

    Args:
        df: Raw movies DataFrame.

    Returns:
        Dict with structure:
        {
            "genres": {"exploded": DataFrame, "lookup": DataFrame},
            "spoken_languages": {"exploded": DataFrame, "lookup": DataFrame},
            ...
        }
    """
    result = {}
    for col, config in JSON_COLUMNS_CONFIG.items():
        exploded, lookup = explode_json_column(df, col, config["keys"])
        result[col] = {"exploded": exploded, "lookup": lookup}
        print(f"  {col}: {len(lookup)} unique items, {len(exploded)} junction rows")
    return result


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: select columns and handle missing values.

    Args:
        df: Raw movies DataFrame loaded from CSV.

    Returns:
        Cleaned DataFrame ready for embedding and insertion.
    """
    df_clean = select_columns(df)
    df_clean = handle_zero_values(df_clean)
    df_clean = handle_release_date(df_clean)
    print(f"Preprocessed {len(df_clean)} rows with {len(df_clean.columns)} columns")
    return df_clean
