"""Compatibility imports for the renamed contextual-state API."""

from src.contextual_state_vectors import (
    get_contextual_positions,
    get_contextual_state_vectors,
)

# Kept for callers of the original API. Contextual states are not token
# embeddings, so new code should use get_contextual_state_vectors.
get_token_state_vectors = get_contextual_state_vectors

__all__ = [
    "get_contextual_positions",
    "get_contextual_state_vectors",
    "get_token_state_vectors",
]
