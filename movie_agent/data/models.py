"""Pydantic data models for Movie."""

from pydantic import BaseModel
from typing import Optional
from datetime import date


class MovieSimple(BaseModel):
    """Model representing a row in the movies_simple table."""

    id: int
    budget: Optional[int] = None
    homepage: Optional[str] = None
    original_language: Optional[str] = None
    original_title: Optional[str] = None
    overview: Optional[str] = None
    popularity: Optional[float] = None
    release_date: Optional[date] = None
    revenue: Optional[int] = None
    runtime: Optional[float] = None
    status: Optional[str] = None
    tagline: Optional[str] = None
    title: str
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    overview_embedding: Optional[list[float]] = None
