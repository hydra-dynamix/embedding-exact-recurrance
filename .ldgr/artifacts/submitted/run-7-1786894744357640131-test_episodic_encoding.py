import unittest
from types import SimpleNamespace

import torch

from src.episodic_encoding import (
    EpisodeAddress,
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

    def test_tau_threshold_produces_the_requested_binary_arrays(self) -> None:
        normalized = torch.tensor([[0.87873, 0.96875, 0.56875, 0.67689]])

        torch.testing.assert_close(
            threshold_state_vectors(normalized, tau_threshold=0.8),
            torch.tensor([[1, 1, 0, 0]], dtype=torch.uint8),
        )
        torch.testing.assert_close(
            threshold_state_vectors(normalized, tau_threshold=0.4),
            torch.tensor([[1, 1, 1, 1]], dtype=torch.uint8),
        )

    def test_encodes_contextual_vectors_directly_without_reference_comparison(self) -> None:
        model = FakeModel(
            torch.tensor([[[2.0, 4.0, 3.0], [99.0, 99.0, 99.0], [5.0, 5.0, 5.0]]])
        )
        encoded = encode_episode(
            model=model,
            input_ids=torch.tensor([[10, 0, 20]]),
            attention_mask=torch.tensor([[1, 0, 1]]),
            tau_threshold=0.6,
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
            encoded.binary_address_sequence,
            torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.uint8),
        )
        torch.testing.assert_close(
            encoded.address.to_tensor(),
            encoded.binary_address_sequence,
        )

    def test_address_identity_preserves_shape_and_bit_order(self) -> None:
        flat_address = EpisodeAddress.from_binary_sequence(
            torch.tensor([[1, 0, 0, 1]], dtype=torch.uint8)
        )
        reshaped_address = EpisodeAddress.from_binary_sequence(
            torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
        )
        reordered_address = EpisodeAddress.from_binary_sequence(
            torch.tensor([[1, 0, 1, 0]], dtype=torch.uint8)
        )

        self.assertNotEqual(flat_address, reshaped_address)
        self.assertNotEqual(flat_address, reordered_address)

    def test_store_retrieves_exact_addresses_without_overwriting_collisions(self) -> None:
        address = EpisodeAddress.from_binary_sequence(
            torch.tensor([[1, 0, 1]], dtype=torch.uint8)
        )
        missing_address = EpisodeAddress.from_binary_sequence(
            torch.tensor([[0, 0, 0]], dtype=torch.uint8)
        )
        store = EpisodicMemoryStore()

        first = store.store(episode_id="episode-a", text="dog", address=address)
        second = store.store(episode_id="episode-b", text="hound", address=address)

        self.assertEqual(store.retrieve(address), (first, second))
        self.assertEqual(store.retrieve(missing_address), ())
        self.assertEqual(store.retrieve_by_id("episode-a"), first)
        self.assertEqual(len(store), 2)
        with self.assertRaisesRegex(ValueError, "already stored"):
            store.store(episode_id="episode-a", text="duplicate", address=address)

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
