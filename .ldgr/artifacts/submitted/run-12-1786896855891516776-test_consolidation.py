import unittest

import torch

from src.consolidation import (
    ConstructionCatalog,
    factor_occurrences,
    factor_recurrence_candidate,
    revise_construction,
)
from src.episodic_encoding import EncodedEpisode, EncodingProvenance
from src.episodic_store import EpisodicMemoryStore
from src.recurrence import LocalRecurrenceIndex, OccurrenceSpan, hydrate_occurrence


def evidence(sequence: torch.Tensor, revision: str = "revision") -> EncodedEpisode:
    state_count, dimensions = sequence.shape
    raw = torch.arange(
        state_count * dimensions,
        dtype=torch.float32,
    ).reshape(state_count, dimensions)
    return EncodedEpisode(
        input_bytes=b"episode",
        encoding_provenance=EncodingProvenance(
            model_id="model",
            model_revision=revision,
            tokenizer_id="tokenizer",
            tokenizer_revision="tokenizer-revision",
            hidden_layer=-1,
            tau_threshold=0.8,
        ),
        input_ids=torch.arange(state_count).reshape(1, state_count),
        attention_mask=torch.ones((1, state_count), dtype=torch.int64),
        contextual_positions=torch.arange(state_count),
        raw_contextual_state_vectors=raw,
        normalized_contextual_state_vectors=raw / max(raw.max().item(), 1.0),
        binary_state_sequence=sequence,
    )


def store_with_sequences(*sequences: torch.Tensor) -> EpisodicMemoryStore:
    store = EpisodicMemoryStore()
    for index, sequence in enumerate(sequences):
        store.store(
            episode_id=f"episode-{index}",
            text=str(index),
            evidence=evidence(sequence),
        )
    return store


def span(store: EpisodicMemoryStore, episode_id: str, start: int, length: int) -> OccurrenceSpan:
    encoding_id = store.retrieve_by_id(
        episode_id
    ).evidence.encoding_provenance.encoding_id
    return OccurrenceSpan(
        episode_id=episode_id,
        start_offset=start,
        length=length,
        encoding_id=encoding_id,
    )


class ConsolidationTests(unittest.TestCase):
    def test_factors_shared_C_and_lossless_occurrence_residuals(self) -> None:
        first = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.uint8)
        second = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.uint8)
        store = store_with_sequences(first, second)
        occurrences = (
            span(store, "episode-0", 0, 2),
            span(store, "episode-1", 0, 2),
        )

        candidate = factor_occurrences(store, occurrences)

        torch.testing.assert_close(
            candidate.shared_component.to_binary_window(),
            torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=torch.uint8),
        )
        expected_residuals = (
            torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.uint8),
            torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.uint8),
        )
        for binding, expected, occurrence in zip(
            candidate.bindings,
            expected_residuals,
            occurrences,
            strict=True,
        ):
            torch.testing.assert_close(binding.to_tensor(), expected)
            self.assertFalse(
                torch.any(
                    binding.to_tensor()
                    & candidate.shared_component.to_binary_window()
                )
            )
            torch.testing.assert_close(
                candidate.reconstruct(occurrence),
                hydrate_occurrence(store, occurrence),
            )
        self.assertFalse(candidate.admitted)

    def test_exact_recurrence_factors_to_zero_residuals(self) -> None:
        shared = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
        store = store_with_sequences(shared, shared)
        index = LocalRecurrenceIndex(min_length=2, max_length=2)
        index.observe_store(store)
        recurrence = index.candidates()[0]

        candidate = factor_recurrence_candidate(store, recurrence)

        torch.testing.assert_close(
            candidate.shared_component.to_binary_window(),
            shared,
        )
        self.assertTrue(
            all(not torch.any(binding.to_tensor()) for binding in candidate.bindings)
        )
        self.assertEqual(
            candidate.source_recurrence_key_digests,
            (recurrence.key.key_digest,),
        )

    def test_revision_is_new_immutable_version_in_same_family(self) -> None:
        first = torch.tensor([[1, 1]], dtype=torch.uint8)
        second = torch.tensor([[1, 0]], dtype=torch.uint8)
        third = torch.tensor([[0, 1]], dtype=torch.uint8)
        store = store_with_sequences(first, second, third)
        predecessor = factor_occurrences(
            store,
            (
                span(store, "episode-0", 0, 1),
                span(store, "episode-1", 0, 1),
            ),
        )
        predecessor_shared = predecessor.shared_component.to_binary_window().clone()

        successor = revise_construction(
            store,
            predecessor,
            (span(store, "episode-2", 0, 1),),
        )
        catalog = ConstructionCatalog()
        catalog.add(predecessor)
        catalog.add(successor)

        self.assertEqual(successor.family_id, predecessor.family_id)
        self.assertNotEqual(successor.version_id, predecessor.version_id)
        self.assertEqual(
            successor.predecessor_version_id,
            predecessor.version_id,
        )
        torch.testing.assert_close(
            predecessor.shared_component.to_binary_window(),
            predecessor_shared,
        )
        self.assertEqual(
            catalog.family_versions(predecessor.family_id),
            (predecessor, successor),
        )
        for occurrence in successor.occurrences:
            torch.testing.assert_close(
                successor.reconstruct(occurrence),
                hydrate_occurrence(store, occurrence),
            )

    def test_rejects_mixed_encoding_bindings(self) -> None:
        store = EpisodicMemoryStore()
        store.store(
            episode_id="episode-a",
            text="a",
            evidence=evidence(torch.tensor([[1]], dtype=torch.uint8), "a"),
        )
        store.store(
            episode_id="episode-b",
            text="b",
            evidence=evidence(torch.tensor([[1]], dtype=torch.uint8), "b"),
        )
        with self.assertRaisesRegex(ValueError, "different encodings"):
            factor_occurrences(
                store,
                (
                    span(store, "episode-a", 0, 1),
                    span(store, "episode-b", 0, 1),
                ),
            )


if __name__ == "__main__":
    unittest.main()
