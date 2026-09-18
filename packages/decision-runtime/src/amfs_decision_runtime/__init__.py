"""Open inference implementation. Training and tenant operations live in SaaS."""
from .model import DecisionScorer, ScorerConfig

__all__ = ["DecisionScorer", "ScorerConfig"]
