import sys
from typing import Any, TextIO

from src.detector import DetectorBank, RepresentedExperience


def print_population_debug(
    experience: RepresentedExperience,
    tokenizer: Any,
    detector_bank: DetectorBank,
    *,
    strongest_detector_count: int = 5,
    stream: TextIO | None = None,
) -> None:
    """Print the detector population recruited at each contextual position."""
    if strongest_detector_count <= 0:
        raise ValueError("strongest_detector_count must be positive")

    contextual_row_count = experience.contextual_positions.shape[0]
    if experience.contextual_positions.ndim != 1:
        raise ValueError("contextual_positions must have shape [contextual_row_count]")
    if experience.input_ids.ndim != 2 or experience.input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, model_position_count]")
    if experience.raw_contextual_state_vectors.shape[0] != contextual_row_count:
        raise ValueError("raw contextual states must contain one vector per row")
    if experience.detector_margins.ndim != 2:
        raise ValueError(
            "detector_margins must have shape [contextual_row_count, detector_count]"
        )
    if experience.coactivation_states.shape != experience.detector_margins.shape:
        raise ValueError("coactivation_states must match detector_margins")
    if experience.detector_margins.shape[0] != contextual_row_count:
        raise ValueError("detector margins must contain one population per row")

    detector_count = detector_bank.raw_reference_vectors.shape[0]
    if experience.detector_margins.shape[1] != detector_count:
        raise ValueError("detector margins must contain one column per detector")

    output = stream if stream is not None else sys.stdout

    for contextual_row, contextual_position_tensor in enumerate(
        experience.contextual_positions
    ):
        contextual_position = int(contextual_position_tensor.item())
        if not 0 <= contextual_position < experience.input_ids.shape[1]:
            raise ValueError("contextual position is outside the input ID sequence")

        token_id = int(experience.input_ids[0, contextual_position].item())
        decoded_token = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        raw_vector_norm = float(
            experience.raw_contextual_state_vectors[contextual_row].norm().item()
        )

        firing_detector_ids = experience.coactivation_states[
            contextual_row
        ].nonzero(as_tuple=False).flatten().tolist()
        ranked_firing_detector_ids = sorted(
            firing_detector_ids,
            key=lambda detector_id: (
                -float(
                    experience.detector_margins[
                        contextual_row,
                        detector_id,
                    ].item()
                ),
                detector_id,
            ),
        )[:strongest_detector_count]

        print(
            f"position={contextual_position} "
            f"token={decoded_token!r} "
            f"raw_vector_norm={raw_vector_norm:.6f} "
            f"firing_detector_count={len(firing_detector_ids)}",
            file=output,
        )

        if not ranked_firing_detector_ids:
            print("  strongest_firing_detectors: none", file=output)
            continue

        print("  strongest_firing_detectors:", file=output)
        for detector_id in ranked_firing_detector_ids:
            margin = float(
                experience.detector_margins[
                    contextual_row,
                    detector_id,
                ].item()
            )
            source_episode_id = detector_bank.source_episode_ids[detector_id]
            source_offset = int(detector_bank.source_offsets[detector_id].item())
            print(
                f"    detector_id={detector_id} "
                f"margin={margin:.6f} "
                f"source_episode={source_episode_id!r} "
                f"source_offset={source_offset}",
                file=output,
            )
