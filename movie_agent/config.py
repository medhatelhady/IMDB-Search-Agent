"""Application configuration and constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://movie_agent:movie_agent_pass@localhost:5432/movie_agent"
)

# Data paths
MOVIES_CSV = os.getenv("MOVIES_CSV", str(PROJECT_ROOT / "data" / "tmdb_5000_movies.csv"))
CREDITS_CSV = os.getenv("CREDITS_CSV", str(PROJECT_ROOT / "data" / "tmdb_5000_credits.csv"))

# Embedding settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

# Search settings
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "60"))
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "10"))
