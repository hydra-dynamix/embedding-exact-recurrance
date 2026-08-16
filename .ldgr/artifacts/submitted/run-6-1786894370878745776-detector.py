from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from src.contextual_state_vectors import (
    get_contextual_positions,
    get_contextual_state_vectors,
)


@dataclass
class DetectorBank:
    """Contextual memory vectors used as detector prototypes, with provenance."""

    raw_reference_vectors: torch.Tensor
    source_episode_ids: list[str]
    source_offsets: torch.Tensor

    def __post_init__(self) -> None:
        if self.raw_reference_vectors.ndim != 2:
            raise ValueError(
                "raw_reference_vectors must have shape "
                "[detector_count, representation_dimensions]"
            )

        detector_count = self.raw_reference_vectors.shape[0]
        if len(self.source_episode_ids) != detector_count:
            raise ValueError(
                "source_episode_ids must contain one ID per detector"
            )
        if self.source_offsets.ndim != 1:
            raise ValueError("source_offsets must have shape [detector_count]")
        if self.source_offsets.shape[0] != detector_count:
            raise ValueError(
                "source_offsets must contain one offset per detector"
            )


@dataclass
class RepresentedExperience:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    contextual_positions: torch.Tensor
    raw_contextual_state_vectors: torch.Tensor
    detector_scores: torch.Tensor
    detector_activations: torch.Tensor
    detector_margins: torch.Tensor
    coactivation_states: torch.Tensor

    @property
    def coactivation_fingerprint_sequence(self) -> torch.Tensor:
        """Return one binary detector fingerprint per contextual position."""
        return self.coactivation_states.to(dtype=torch.uint8)


def build_detector_bank(
    memory_state_sequences: Sequence[torch.Tensor],
    source_episode_ids: Sequence[str],
    source_offset_sequences: Sequence[torch.Tensor],
) -> DetectorBank:
    """Build one provenance-linked detector per experienced contextual state."""
    sequence_count = len(memory_state_sequences)
    if sequence_count == 0:
        raise ValueError("memory_state_sequences must contain at least one tensor")
    if len(source_episode_ids) != sequence_count:
        raise ValueError("source_episode_ids must contain one ID per sequence")
    if len(source_offset_sequences) != sequence_count:
        raise ValueError(
            "source_offset_sequences must contain one tensor per sequence"
        )

    detector_source_episode_ids: list[str] = []
    validated_offsets: list[torch.Tensor] = []

    for sequence_index, (state_vectors, offsets) in enumerate(
        zip(memory_state_sequences, source_offset_sequences, strict=True)
    ):
        if state_vectors.ndim != 2:
            raise ValueError(
                f"memory_state_sequences[{sequence_index}] must have shape "
                "[token_count, representation_dimensions]"
            )
        if offsets.ndim != 1 or offsets.shape[0] != state_vectors.shape[0]:
            raise ValueError(
                f"source_offset_sequences[{sequence_index}] must contain "
                "one offset per contextual state"
            )

        detector_source_episode_ids.extend(
            [source_episode_ids[sequence_index]] * state_vectors.shape[0]
        )
        validated_offsets.append(offsets.detach().cpu().long())

    raw_detector_reference_vectors = torch.cat(
        tuple(memory_state_sequences),
        dim=0,
    ).float()
    detector_source_offsets = torch.cat(validated_offsets, dim=0)

    return DetectorBank(
        raw_reference_vectors=raw_detector_reference_vectors,
        source_episode_ids=detector_source_episode_ids,
        source_offsets=detector_source_offsets,
    )


def represent_experience(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    detector_bank: DetectorBank,
    tau_threshold: float = 0.95,
) -> RepresentedExperience:
    """Represent each contextual position as a full detector population state."""
    if not 0.0 <= tau_threshold <= 1.0:
        raise ValueError("tau_threshold must be between 0 and 1")

    raw_detector_reference_vectors = detector_bank.raw_reference_vectors
    scoring_reference_vectors = raw_detector_reference_vectors.to(
        dtype=torch.float32,
    )

    raw_contextual_state_vectors = get_contextual_state_vectors(
        model,
        input_ids,
        attention_mask,
    ).to(
        device=raw_detector_reference_vectors.device,
        dtype=torch.float32,
    )

    if raw_contextual_state_vectors.ndim != 2:
        raise ValueError(
            "contextual state vectors must have shape "
            "[contextual_position_count, representation_dimensions]"
        )
    if (
        raw_contextual_state_vectors.shape[1]
        != raw_detector_reference_vectors.shape[1]
    ):
        raise ValueError(
            "contextual states and detector references must have the same "
            "representation dimension"
        )

    detector_scores = raw_contextual_state_vectors @ scoring_reference_vectors.T
    normalized_contextual_states = functional.normalize(
        raw_contextual_state_vectors,
        p=2,
        dim=1,
    )
    normalized_detector_references = functional.normalize(
        scoring_reference_vectors,
        p=2,
        dim=1,
    )
    cosine_scores = normalized_contextual_states @ normalized_detector_references.T
    detector_activations = ((cosine_scores + 1.0) / 2.0).clamp(0.0, 1.0)
    detector_margins = detector_activations - tau_threshold
    coactivation_states = detector_activations >= tau_threshold

    expected_score_shape = (
        raw_contextual_state_vectors.shape[0],
        raw_detector_reference_vectors.shape[0],
    )
    assert detector_scores.shape == expected_score_shape
    assert detector_activations.shape == detector_scores.shape
    assert detector_margins.shape == detector_scores.shape
    assert coactivation_states.shape == detector_scores.shape

    contextual_positions = get_contextual_positions(attention_mask)
    if contextual_positions.shape[0] != raw_contextual_state_vectors.shape[0]:
        raise ValueError(
            "contextual_positions must contain one source position per "
            "contextual state"
        )

    return RepresentedExperience(
        input_ids=input_ids,
        attention_mask=attention_mask,
        contextual_positions=contextual_positions,
        raw_contextual_state_vectors=raw_contextual_state_vectors,
        detector_scores=detector_scores,
        detector_activations=detector_activations,
        detector_margins=detector_margins,
        coactivation_states=coactivation_states,
    )
