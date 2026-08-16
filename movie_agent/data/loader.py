"""Load and join TMDB CSV files."""

import pandas as pd
from movie_agent.config import MOVIES_CSV, CREDITS_CSV


def load_movies(path: str = None) -> pd.DataFrame:
    """Load the TMDB 5000 movies CSV."""
    csv_path = path or MOVIES_CSV
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} movies from {csv_path}")
    return df


def load_credits(path: str = None) -> pd.DataFrame:
    """Load the TMDB 5000 credits CSV."""
    csv_path = path or CREDITS_CSV
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} credit records from {csv_path}")
    return df


def load_all(movies_path: str = None, credits_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs and return (movies_df, credits_df)."""
    movies_df = load_movies(movies_path)
    credits_df = load_credits(credits_path)
    return movies_df, credits_df
