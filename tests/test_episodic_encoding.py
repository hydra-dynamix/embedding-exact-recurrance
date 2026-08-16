import unittest
from types import SimpleNamespace

import torch

from src.episodic_encoding import (
    EncodedEpisode,
    EncodingProvenance,
    SequencedBinaryAddress,
    encode_episode,
    normalize_contextual_state_vectors,
    threshold_state_vectors,
)
from src.episodic_store import EpisodicMemoryStore


class FakeModel(torch.nn.Module):
    def __init__(self, final_layer_states: torch.Tensor) -> None:
        super().__init__()
        self.final_layer_states = final_layer_states

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        if not output_hidden_states:
            raise AssertionError("hidden states must be requested")
        return SimpleNamespace(hidden_states=(self.final_layer_states,))


def encoding_provenance(tau_threshold: float = 0.6) -> EncodingProvenance:
    return EncodingProvenance(
        model_id="test-model",
        model_revision="test-model-revision",
        tokenizer_id="test-tokenizer",
        tokenizer_revision="test-tokenizer-revision",
        hidden_layer=-1,
        tau_threshold=tau_threshold,
    )


def encoded_evidence(binary_state_sequence: torch.Tensor) -> EncodedEpisode:
    state_count, state_dimensions = binary_state_sequence.shape
    raw = torch.arange(
        state_count * state_dimensions,
        dtype=torch.float32,
    ).reshape(state_count, state_dimensions)
    return EncodedEpisode(
        input_bytes=b"synthetic episode",
        encoding_provenance=encoding_provenance(),
        input_ids=torch.arange(state_count).reshape(1, state_count),
        attention_mask=torch.ones((1, state_count), dtype=torch.int64),
        contextual_positions=torch.arange(state_count),
        raw_contextual_state_vectors=raw,
        normalized_contextual_state_vectors=raw / max(raw.max().item(), 1.0),
        binary_state_sequence=binary_state_sequence,
    )


class EpisodicEncodingTests(unittest.TestCase):
    def test_normalizes_each_vector_independently_and_handles_constants(self) -> None:
        normalized = normalize_contextual_state_vectors(
            torch.tensor([[2.0, 4.0, 3.0], [5.0, 5.0, 5.0]])
        )

        torch.testing.assert_close(
            normalized,
            torch.tensor([[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]),
        )
        self.assertGreaterEqual(normalized.min().item(), 0.0)
        self.assertLessEqual(normalized.max().item(), 1.0)

    def test_tau_threshold_produces_individual_binary_states_B_t(self) -> None:
        normalized = torch.tensor([[0.87873, 0.96875, 0.56875, 0.67689]])

        torch.testing.assert_close(
            threshold_state_vectors(normalized, tau_threshold=0.8),
            torch.tensor([[1, 1, 0, 0]], dtype=torch.uint8),
        )
        torch.testing.assert_close(
            threshold_state_vectors(normalized, tau_threshold=0.4),
            torch.tensor([[1, 1, 1, 1]], dtype=torch.uint8),
        )

    def test_encodes_B_states_and_collects_them_into_sequenced_address(self) -> None:
        model = FakeModel(
            torch.tensor(
                [[[2.0, 4.0, 3.0], [99.0, 99.0, 99.0], [5.0, 5.0, 5.0]]]
            )
        )
        encoded = encode_episode(
            model=model,
            input_ids=torch.tensor([[10, 0, 20]]),
            attention_mask=torch.tensor([[1, 0, 1]]),
            input_bytes=b"synthetic episode",
            encoding_provenance=encoding_provenance(),
        )

        torch.testing.assert_close(encoded.contextual_positions, torch.tensor([0, 2]))
        torch.testing.assert_close(
            encoded.raw_contextual_state_vectors,
            torch.tensor([[2.0, 4.0, 3.0], [5.0, 5.0, 5.0]]),
        )
        torch.testing.assert_close(
            encoded.normalized_contextual_state_vectors,
            torch.tensor([[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            encoded.binary_state_sequence,
            torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.uint8),
        )
        torch.testing.assert_close(
            encoded.sequenced_address.to_binary_state_sequence(),
            encoded.binary_state_sequence,
        )

    def test_sequenced_address_preserves_state_count_dimensions_and_order(self) -> None:
        flat_address = SequencedBinaryAddress.from_binary_state_sequence(
            torch.tensor([[1, 0, 0, 1]], dtype=torch.uint8)
        )
        sequenced_address = SequencedBinaryAddress.from_binary_state_sequence(
            torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
        )
        reordered_address = SequencedBinaryAddress.from_binary_state_sequence(
            torch.tensor([[1, 0, 1, 0]], dtype=torch.uint8)
        )

        self.assertNotEqual(flat_address, sequenced_address)
        self.assertNotEqual(flat_address, reordered_address)

    def test_rejects_malformed_sequenced_address_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            SequencedBinaryAddress(
                state_count=2,
                state_dimensions=2,
                binary_state_values=b"\x01",
            )
        with self.assertRaisesRegex(ValueError, "only zero and one"):
            SequencedBinaryAddress(
                state_count=1,
                state_dimensions=1,
                binary_state_values=b"\x02",
            )

    def test_store_keys_complete_evidence_by_sequenced_address(self) -> None:
        binary_state_sequence = torch.tensor([[1, 0, 1]], dtype=torch.uint8)
        first_evidence = encoded_evidence(binary_state_sequence.clone())
        second_evidence = encoded_evidence(binary_state_sequence.clone())
        missing_address = SequencedBinaryAddress.from_binary_state_sequence(
            torch.tensor([[0, 0, 0]], dtype=torch.uint8)
        )
        sequenced_address = first_evidence.sequenced_address
        store = EpisodicMemoryStore()

        first = store.store(
            episode_id="episode-a",
            text="dog",
            evidence=first_evidence,
        )
        second = store.store(
            episode_id="episode-b",
            text="hound",
            evidence=second_evidence,
        )
        first_evidence.binary_state_sequence.zero_()

        self.assertEqual(store.retrieve(sequenced_address), (first, second))
        self.assertEqual(store.retrieve(missing_address), ())
        self.assertEqual(store.retrieve_by_id("episode-a"), first)
        torch.testing.assert_close(
            first.evidence.binary_state_sequence,
            binary_state_sequence,
        )
        torch.testing.assert_close(
            first.evidence.raw_contextual_state_vectors,
            encoded_evidence(binary_state_sequence).raw_contextual_state_vectors,
        )
        self.assertEqual(first.sequenced_address, sequenced_address)
        self.assertEqual(len(store), 2)
        with self.assertRaisesRegex(ValueError, "already stored"):
            store.store(
                episode_id="episode-a",
                text="duplicate",
                evidence=encoded_evidence(binary_state_sequence),
            )

    def test_rejects_invalid_thresholds_and_nonfinite_vectors(self) -> None:
        for invalid_threshold in (-0.01, 1.01, float("nan")):
            with self.subTest(tau_threshold=invalid_threshold):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    threshold_state_vectors(
                        torch.tensor([[0.5]]),
                        tau_threshold=invalid_threshold,
                    )

        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_contextual_state_vectors(torch.tensor([[float("nan")]]))


if __name__ == "__main__":
    unittest.main()
