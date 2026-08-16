from dataclasses import dataclass

import torch

from src.contextual_state_vectors import (
    get_contextual_positions,
    get_contextual_state_vectors,
)


@dataclass(frozen=True)
class SequencedBinaryAddress:
    """A shape-aware episode address containing the ordered sequence of B states."""

    state_count: int
    state_dimensions: int
    binary_state_values: bytes

    def __post_init__(self) -> None:
        if self.state_count <= 0:
            raise ValueError("state_count must be positive")
        if self.state_dimensions <= 0:
            raise ValueError("state_dimensions must be positive")
        expected_value_count = self.state_count * self.state_dimensions
        if len(self.binary_state_values) != expected_value_count:
            raise ValueError(
                "binary_state_values must contain exactly "
                "state_count * state_dimensions values"
            )
        if any(value not in (0, 1) for value in self.binary_state_values):
            raise ValueError("binary_state_values must contain only zero and one")

    @classmethod
    def from_binary_state_sequence(
        cls,
        binary_state_sequence: torch.Tensor,
    ) -> "SequencedBinaryAddress":
        """Create A_e from an ordered matrix whose rows are binary states B_t."""
        if binary_state_sequence.ndim != 2:
            raise ValueError(
                "binary_state_sequence must have shape "
                "[state_count, state_dimensions]"
            )
        if binary_state_sequence.numel() == 0:
            raise ValueError("binary_state_sequence must not be empty")
        if not torch.all(
            (binary_state_sequence == 0) | (binary_state_sequence == 1)
        ):
            raise ValueError(
                "binary_state_sequence must contain only zero and one"
            )

        binary_state_values = bytes(
            binary_state_sequence.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            .flatten()
            .tolist()
        )
        return cls(
            state_count=binary_state_sequence.shape[0],
            state_dimensions=binary_state_sequence.shape[1],
            binary_state_values=binary_state_values,
        )

    def to_binary_state_sequence(self) -> torch.Tensor:
        """Reconstruct A_e as a uint8 matrix with one B_t per row."""
        return torch.tensor(
            tuple(self.binary_state_values),
            dtype=torch.uint8,
        ).reshape(self.state_count, self.state_dimensions)


@dataclass
class EncodedEpisode:
    """Complete in-process evidence for H -> N -> B and its sequence A_e."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    contextual_positions: torch.Tensor
    raw_contextual_state_vectors: torch.Tensor
    normalized_contextual_state_vectors: torch.Tensor
    binary_state_sequence: torch.Tensor

    @property
    def sequenced_address(self) -> SequencedBinaryAddress:
        """Return A_e = (B_0, ..., B_(T-1))."""
        return SequencedBinaryAddress.from_binary_state_sequence(
            self.binary_state_sequence
        )


def normalize_contextual_state_vectors(
    raw_contextual_state_vectors: torch.Tensor,
) -> torch.Tensor:
    """Min-max normalize each contextual vector H_t independently into N_t."""
    if raw_contextual_state_vectors.ndim != 2:
        raise ValueError(
            "raw_contextual_state_vectors must have shape "
            "[state_count, state_dimensions]"
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
    safe_ranges = torch.where(
        nonconstant_rows,
        row_ranges,
        torch.ones_like(row_ranges),
    )
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
    """Threshold each N_t into a uint8 binary state B_t."""
    if not 0.0 <= tau_threshold <= 1.0:
        raise ValueError("tau_threshold must be between 0 and 1")
    if normalized_contextual_state_vectors.ndim != 2:
        raise ValueError(
            "normalized_contextual_state_vectors must have shape "
            "[state_count, state_dimensions]"
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
    """Encode one episode as H, N, and the ordered sequence A_e of B states."""
    raw_contextual_state_vectors = get_contextual_state_vectors(
        model,
        input_ids,
        attention_mask,
    )
    normalized_contextual_state_vectors = normalize_contextual_state_vectors(
        raw_contextual_state_vectors
    )
    binary_state_sequence = threshold_state_vectors(
        normalized_contextual_state_vectors,
        tau_threshold,
    )
    contextual_positions = get_contextual_positions(attention_mask)

    if contextual_positions.shape[0] != binary_state_sequence.shape[0]:
        raise ValueError(
            "contextual positions and binary states must have equal length"
        )

    return EncodedEpisode(
        input_ids=input_ids,
        attention_mask=attention_mask,
        contextual_positions=contextual_positions,
        raw_contextual_state_vectors=raw_contextual_state_vectors,
        normalized_contextual_state_vectors=normalized_contextual_state_vectors,
        binary_state_sequence=binary_state_sequence,
    )
