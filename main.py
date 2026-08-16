import argparse
from pathlib import Path

from src.consolidation import factor_recurrence_candidate
from src.episodic_encoding import EncodingProvenance, encode_episode
from src.episodic_store import EpisodicMemoryStore
from src.persistent_store import PersistentEpisodicMemoryStore
from src.recurrence import LocalRecurrenceIndex


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "texts",
        nargs="*",
        default=["dog"],
        help="Text blocks to encode, store, and retrieve (default: dog).",
    )
    parser.add_argument(
        "--tau-threshold",
        type=float,
        default=0.8,
        help="Embedding-component threshold between 0 and 1 (default: 0.8).",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        help="Optional directory for uncompressed restartable episode evidence.",
    )
    parser.add_argument(
        "--debug-encoding",
        action="store_true",
        help="Print raw and normalized contextual vectors.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.tau_threshold <= 1.0:
        parser.error("--tau-threshold must be between 0 and 1")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model.eval()
    model_device = next(model.parameters()).device
    episode_store = (
        PersistentEpisodicMemoryStore(args.store_root)
        if args.store_root is not None
        else EpisodicMemoryStore()
    )
    provenance = EncodingProvenance(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        hidden_layer=-1,
        tau_threshold=args.tau_threshold,
    )
    first_episode_number = len(episode_store) + 1

    for episode_index, text in enumerate(
        args.texts,
        start=first_episode_number,
    ):
        model_inputs = tokenizer(text, return_tensors="pt").to(model_device)
        encoded_episode = encode_episode(
            model=model,
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            input_bytes=text.encode("utf-8"),
            encoding_provenance=provenance,
        )
        episode_id = f"episode-{episode_index:03d}"
        stored_episode = episode_store.store(
            episode_id=episode_id,
            text=text,
            evidence=encoded_episode,
        )
        retrieved_episodes = episode_store.retrieve(
            encoded_episode.sequenced_address
        )

        print(f"episode_id={episode_id!r} text={text!r}")
        print(
            "binary_state_sequence_shape:",
            tuple(encoded_episode.binary_state_sequence.shape),
        )
        print(
            "binary_state_sequence:",
            encoded_episode.binary_state_sequence.tolist(),
        )
        print(
            "sequenced_address:",
            {
                "state_count": stored_episode.sequenced_address.state_count,
                "state_dimensions": (
                    stored_episode.sequenced_address.state_dimensions
                ),
            },
        )
        print(
            "retrieved_episode_ids:",
            [episode.episode_id for episode in retrieved_episodes],
        )

        if args.debug_encoding:
            print(
                "raw_contextual_state_vectors:",
                encoded_episode.raw_contextual_state_vectors.tolist(),
            )
            print(
                "normalized_contextual_state_vectors:",
                encoded_episode.normalized_contextual_state_vectors.tolist(),
            )

    recurrence_index = LocalRecurrenceIndex()
    indexed_windows = recurrence_index.observe_store(episode_store)
    recurrence_candidates = recurrence_index.candidates()
    construction_candidates = tuple(
        factor_recurrence_candidate(episode_store, recurrence)
        for recurrence in recurrence_candidates
    )
    print("indexed_local_windows:", indexed_windows)
    print("recurrent_candidate_count:", len(recurrence_candidates))
    for recurrence, construction in zip(
        recurrence_candidates,
        construction_candidates,
        strict=True,
    ):
        print(
            "recurrence_candidate:",
            {
                "key_digest": recurrence.key.key_digest,
                "length": recurrence.key.length,
                "occurrences": [
                    {
                        "episode_id": item.episode_id,
                        "start_offset": item.start_offset,
                    }
                    for item in recurrence.occurrences
                ],
                "construction_version_id": construction.version_id,
                "admitted": construction.admitted,
            },
        )
