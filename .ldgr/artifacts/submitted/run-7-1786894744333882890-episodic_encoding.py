from dataclasses import dataclass

import torch

from src.contextual_state_vectors import (
    get_contextual_positions,
    get_contextual_state_vectors,
)


@dataclass(frozen=True)
class EpisodeAddress:
    """An exact, shape-aware address made from an ordered binary state sequence."""

    contextual_position_count: int
    representation_dimensions: int
    binary_values: bytes

    @classmethod
    def from_binary_sequence(cls, binary_sequence: torch.Tensor) -> "EpisodeAddress":
        if binary_sequence.ndim != 2:
            raise ValueError(
                "binary_sequence must have shape "
                "[contextual_position_count, representation_dimensions]"
            )
        if binary_sequence.numel() == 0:
            raise ValueError("binary_sequence must not be empty")
        if not torch.all((binary_sequence == 0) | (binary_sequence == 1)):
            raise ValueError("binary_sequence must contain only zero and one")

        binary_values = bytes(
            binary_sequence.detach().to(device="cpu", dtype=torch.uint8)
            .contiguous()
            .flatten()
            .tolist()
        )
        return cls(
            contextual_position_count=binary_sequence.shape[0],
            representation_dimensions=binary_sequence.shape[1],
            binary_values=binary_values,
        )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the address as a uint8 matrix."""
        return torch.tensor(tuple(self.binary_values), dtype=torch.uint8).reshape(
            self.contextual_position_count,
            self.representation_dimensions,
        )


@dataclass
class EncodedEpisode:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    contextual_positions: torch.Tensor
    raw_contextual_state_vectors: torch.Tensor
    normalized_contextual_state_vectors: torch.Tensor
    binary_address_sequence: torch.Tensor

    @property
    def address(self) -> EpisodeAddress:
        return EpisodeAddress.from_binary_sequence(self.binary_address_sequence)


def normalize_contextual_state_vectors(
    raw_contextual_state_vectors: torch.Tensor,
) -> torch.Tensor:
    """Min-max normalize each contextual vector independently into [0, 1]."""
    if raw_contextual_state_vectors.ndim != 2:
        raise ValueError(
            "raw_contextual_state_vectors must have shape "
            "[contextual_position_count, representation_dimensions]"
        )
    if raw_contextual_state_vectors.numel() == 0:
        raise ValueError("raw_contextual_state_vectors must not be empty")

    state_vectors = raw_contextual_state_vectors.to(dtype=torch.float32)
    if not torch.isfinite(state_vectors).all():
        raise ValueError("raw_contextual_state_vectors must contain finite values")

    row_minimums = state_vectors.amin(dim=1, keepdim=True)
    row_maximums = state_vectors.amax(dim=1, keepdim=True)
    row_ranges = row_maximums - row_minimums
    nonconstant_rows = row_ranges > 0
    safe_ranges = torch.where(nonconstant_rows, row_ranges, torch.ones_like(row_ranges))
    normalized_states = (state_vectors - row_minimums) / safe_ranges
    normalized_states = torch.where(
        nonconstant_rows,
        normalized_states,
        torch.zeros_like(normalized_states),
    )
    return normalized_states.clamp(0.0, 1.0)


def threshold_state_vectors(
    normalized_contextual_state_vectors: torch.Tensor,
    tau_threshold: float,
) -> torch.Tensor:
    """Convert bounded contextual vectors into an ordered uint8 binary sequence."""
    if not 0.0 <= tau_threshold <= 1.0:
        raise ValueError("tau_threshold must be between 0 and 1")
    if normalized_contextual_state_vectors.ndim != 2:
        raise ValueError(
            "normalized_contextual_state_vectors must have shape "
            "[contextual_position_count, representation_dimensions]"
        )
    if normalized_contextual_state_vectors.numel() == 0:
        raise ValueError("normalized_contextual_state_vectors must not be empty")
    if not torch.isfinite(normalized_contextual_state_vectors).all():
        raise ValueError(
            "normalized_contextual_state_vectors must contain finite values"
        )
    if not torch.all(
        (normalized_contextual_state_vectors >= 0)
        & (normalized_contextual_state_vectors <= 1)
    ):
        raise ValueError(
            "normalized_contextual_state_vectors must be between 0 and 1"
        )

    return (normalized_contextual_state_vectors >= tau_threshold).to(torch.uint8)


def encode_episode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tau_threshold: float,
) -> EncodedEpisode:
    """Encode a text block as an ordered sequence of binary embedding states."""
    raw_contextual_state_vectors = get_contextual_state_vectors(
        model,
        input_ids,
        attention_mask,
    )
    normalized_contextual_state_vectors = normalize_contextual_state_vectors(
        raw_contextual_state_vectors
    )
    binary_address_sequence = threshold_state_vectors(
        normalized_contextual_state_vectors,
        tau_threshold,
    )
    contextual_positions = get_contextual_positions(attention_mask)

    if contextual_positions.shape[0] != binary_address_sequence.shape[0]:
        raise ValueError(
            "contextual positions and binary address rows must have equal length"
        )

    return EncodedEpisode(
        input_ids=input_ids,
        attention_mask=attention_mask,
        contextual_positions=contextual_positions,
        raw_contextual_state_vectors=raw_contextual_state_vectors,
        normalized_contextual_state_vectors=normalized_contextual_state_vectors,
        binary_address_sequence=binary_address_sequence,
    )
