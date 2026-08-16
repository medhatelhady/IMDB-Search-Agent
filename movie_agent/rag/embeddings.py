"""Embedding model setup (sentence-transformers).

The model is loaded once at module import time so subsequent calls
to encode_texts() and encode_single() are fast.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from movie_agent.config import EMBEDDING_MODEL, PROJECT_ROOT

# Local directory to save/load the model
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / EMBEDDING_MODEL


def _load_model() -> SentenceTransformer:
    """Load the model from local cache. Downloads and saves if not found."""
    if MODEL_CACHE_DIR.exists():
        print(f"Loading embedding model from local: {MODEL_CACHE_DIR}")
        m = SentenceTransformer(str(MODEL_CACHE_DIR))
    else:
        print(f"Local model not found. Downloading '{EMBEDDING_MODEL}'...")
        m = SentenceTransformer(EMBEDDING_MODEL)
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        m.save(str(MODEL_CACHE_DIR))
        print(f"Model saved locally to {MODEL_CACHE_DIR}")
    print("Model loaded.")
    return m


# Load the model once at import time
model = _load_model()


def save_model():
    """Download the model and save it locally."""
    print(f"Downloading model '{EMBEDDING_MODEL}' and saving to {MODEL_CACHE_DIR}")
    m = SentenceTransformer(EMBEDDING_MODEL)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(MODEL_CACHE_DIR))
    print(f"Model saved to {MODEL_CACHE_DIR}")


def encode_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for encoding.
        show_progress: Whether to show a progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings


def encode_single(text: str) -> list[float]:
    """Generate embedding for a single text. Returns a list of floats."""
    embedding = model.encode(text)
    return embedding.tolist()
