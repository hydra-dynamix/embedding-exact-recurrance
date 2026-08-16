import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.episodic_encoding import EncodedEpisode, EncodingProvenance
from src.persistent_store import PersistentEpisodicMemoryStore


def evidence(
    binary_state_sequence: torch.Tensor,
    *,
    tau_threshold: float = 0.8,
    input_bytes: bytes = b"episode bytes",
) -> EncodedEpisode:
    state_count, dimensions = binary_state_sequence.shape
    raw = torch.arange(
        state_count * dimensions,
        dtype=torch.float32,
    ).reshape(state_count, dimensions)
    provenance = EncodingProvenance(
        model_id="test-model",
        model_revision="model-revision",
        tokenizer_id="test-tokenizer",
        tokenizer_revision="tokenizer-revision",
        hidden_layer=-1,
        tau_threshold=tau_threshold,
    )
    return EncodedEpisode(
        input_bytes=input_bytes,
        encoding_provenance=provenance,
        input_ids=torch.arange(state_count).reshape(1, state_count),
        attention_mask=torch.ones((1, state_count), dtype=torch.int64),
        contextual_positions=torch.arange(state_count),
        raw_contextual_state_vectors=raw,
        normalized_contextual_state_vectors=raw / max(raw.max().item(), 1.0),
        binary_state_sequence=binary_state_sequence,
    )


class PersistentStoreTests(unittest.TestCase):
    def test_restart_hydrates_complete_collision_bucket_under_A_e(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.uint8)
            first_evidence = evidence(binary, input_bytes=b"first bytes")
            second_evidence = evidence(binary, input_bytes=b"second bytes")
            address = first_evidence.sequenced_address
            store = PersistentEpisodicMemoryStore(root)

            store.store(
                episode_id="episode-a",
                text="first",
                evidence=first_evidence,
            )
            store.store(
                episode_id="episode-b",
                text="second",
                evidence=second_evidence,
            )

            address_directory = root / "addresses" / address.address_digest
            self.assertTrue((address_directory / "address.json").is_file())
            self.assertEqual(
                (address_directory / "binary-state-values.u8.raw").read_bytes(),
                address.binary_state_values,
            )

            restarted = PersistentEpisodicMemoryStore(root)
            occurrences = restarted.retrieve(address)
            self.assertEqual(
                [item.episode_id for item in occurrences],
                ["episode-a", "episode-b"],
            )
            self.assertEqual(
                restarted.retrieve_by_id("episode-a").evidence.input_bytes,
                b"first bytes",
            )
            torch.testing.assert_close(
                restarted.retrieve_by_id("episode-a").evidence.raw_contextual_state_vectors,
                first_evidence.raw_contextual_state_vectors,
            )
            torch.testing.assert_close(
                restarted.retrieve_by_id("episode-b").evidence.binary_state_sequence,
                binary,
            )
            self.assertEqual(
                restarted.retrieve_by_id("episode-a").evidence.encoding_provenance,
                first_evidence.encoding_provenance,
            )

    def test_tampered_tensor_fails_hash_validated_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = PersistentEpisodicMemoryStore(root)
            stored = store.store(
                episode_id="episode-a",
                text="first",
                evidence=evidence(torch.tensor([[1, 0]], dtype=torch.uint8)),
            )
            pointer_path = next((root / "episode-index").glob("*.json"))
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            metadata_path = root / pointer["metadata"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            binary_path = metadata_path.parent / metadata["tensors"][
                "binary_state_sequence"
            ]["path"]
            binary_path.write_bytes(binary_path.read_bytes() + b"tamper")

            with self.assertRaisesRegex(RuntimeError, "byte count changed"):
                PersistentEpisodicMemoryStore(root)
            self.assertEqual(stored.episode_id, "episode-a")

    def test_encoding_identity_changes_with_tau(self) -> None:
        first = evidence(
            torch.tensor([[1]], dtype=torch.uint8),
            tau_threshold=0.8,
        ).encoding_provenance
        second = evidence(
            torch.tensor([[1]], dtype=torch.uint8),
            tau_threshold=0.9,
        ).encoding_provenance
        self.assertNotEqual(first.encoding_id, second.encoding_id)


if __name__ == "__main__":
    unittest.main()
