import argparse

from src.episodic_encoding import encode_episode
from src.episodic_store import EpisodicMemoryStore


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
        "--debug-encoding",
        action="store_true",
        help="Print raw and normalized contextual vectors.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.tau_threshold <= 1.0:
        parser.error("--tau-threshold must be between 0 and 1")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model.eval()
    model_device = next(model.parameters()).device
    episode_store = EpisodicMemoryStore()

    for episode_index, text in enumerate(args.texts, start=1):
        model_inputs = tokenizer(text, return_tensors="pt").to(model_device)
        encoded_episode = encode_episode(
            model=model,
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            tau_threshold=args.tau_threshold,
        )
        episode_id = f"episode-{episode_index:03d}"
        episode_store.store(
            episode_id=episode_id,
            text=text,
            address=encoded_episode.address,
        )
        retrieved_episodes = episode_store.retrieve(encoded_episode.address)

        print(f"episode_id={episode_id!r} text={text!r}")
        print(
            "binary_address_shape:",
            tuple(encoded_episode.binary_address_sequence.shape),
        )
        print(
            "binary_address_sequence:",
            encoded_episode.binary_address_sequence.tolist(),
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
