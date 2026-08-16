from src.debugging import print_population_debug
from src.detector import (
    build_detector_bank,
    calibrate_detector_thresholds,
    represent_experience,
)
from src.contextual_state_vectors import (
    get_contextual_positions,
    get_contextual_state_vectors,
)


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model.eval()
    model_device = next(model.parameters()).device

    memory_episodes = {
        "memory-001": (
            "A detector represents a previously experienced contextual state."
        ),
        "memory-002": (
            "Memory observes the detector population one token position at a time."
        ),
    }
    calibration_texts = [
        "Calibration establishes the normal response level for each detector.",
        "Separate experiences are used to estimate detector thresholds.",
    ]

    memory_state_sequences = []
    memory_offset_sequences = []
    for memory_text in memory_episodes.values():
        memory_inputs = tokenizer(memory_text, return_tensors="pt").to(
            model_device
        )
        memory_state_sequences.append(
            get_contextual_state_vectors(
                model,
                memory_inputs["input_ids"],
                memory_inputs["attention_mask"],
            )
        )
        memory_offset_sequences.append(
            get_contextual_positions(memory_inputs["attention_mask"])
        )

    detector_bank = build_detector_bank(
        memory_state_sequences=memory_state_sequences,
        source_episode_ids=list(memory_episodes),
        source_offset_sequences=memory_offset_sequences,
    )

    calibration_state_sequences = []
    for calibration_text in calibration_texts:
        calibration_inputs = tokenizer(
            calibration_text,
            return_tensors="pt",
        ).to(model_device)
        calibration_state_sequences.append(
            get_contextual_state_vectors(
                model,
                calibration_inputs["input_ids"],
                calibration_inputs["attention_mask"],
            )
        )

    detector_thresholds = calibrate_detector_thresholds(
        calibration_state_sequences,
        detector_bank,
        quantile=0.95,
    )

    experience_texts = [
        "The current experience becomes an ordered sequence of coactivation states.",
        "The detector population changes as each new contextual state arrives.",
    ]

    print(
        "detector_raw_reference_vectors:",
        tuple(detector_bank.raw_reference_vectors.shape),
    )
    print("detector_thresholds:", tuple(detector_thresholds.shape))

    for experience_index, experience_text in enumerate(experience_texts, start=1):
        experience_inputs = tokenizer(
            experience_text,
            return_tensors="pt",
        ).to(model_device)
        experience = represent_experience(
            model=model,
            input_ids=experience_inputs["input_ids"],
            attention_mask=experience_inputs["attention_mask"],
            detector_bank=detector_bank,
            detector_thresholds=detector_thresholds,
        )

        print(f"\nexperience={experience_index} text={experience_text!r}")
        print_population_debug(
            experience=experience,
            tokenizer=tokenizer,
            detector_bank=detector_bank,
        )
