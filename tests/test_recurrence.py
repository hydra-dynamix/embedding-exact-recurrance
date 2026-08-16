from pathlib import Path
import tempfile
import unittest

import torch

from src.episodic_encoding import EncodedEpisode, EncodingProvenance
from src.episodic_store import EpisodicMemoryStore
from src.persistent_store import PersistentEpisodicMemoryStore
from src.recurrence import (
    BinarySubsequenceKey,
    LocalRecurrenceIndex,
    hydrate_occurrence,
)


def evidence(
    sequence: torch.Tensor,
    *,
    revision: str = "revision-a",
) -> EncodedEpisode:
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


def populated_store(store: EpisodicMemoryStore) -> EpisodicMemoryStore:
    store.store(
        episode_id="episode-a",
        text="a",
        evidence=evidence(
            torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.uint8)
        ),
    )
    store.store(
        episode_id="episode-b",
        text="b",
        evidence=evidence(
            torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.uint8)
        ),
    )
    return store


class LocalRecurrenceTests(unittest.TestCase):
    def test_indexes_every_legal_window_and_finds_complete_recurrence(self) -> None:
        store = populated_store(EpisodicMemoryStore())
        index = LocalRecurrenceIndex()

        self.assertEqual(index.observe_store(store), 12)
        self.assertEqual(index.indexed_window_count, 12)
        candidates = index.candidates()
        by_values = {
            tuple(map(tuple, item.key.to_binary_window().tolist())): item
            for item in candidates
        }
        shared = by_values[((1, 0), (0, 1))]
        self.assertEqual(
            [(item.episode_id, item.start_offset) for item in shared.occurrences],
            [("episode-a", 0), ("episode-b", 1)],
        )
        self.assertEqual(shared.distinct_episode_count, 2)
        for occurrence in shared.occurrences:
            torch.testing.assert_close(
                hydrate_occurrence(store, occurrence),
                shared.key.to_binary_window(),
            )

    def test_does_not_create_cross_episode_windows(self) -> None:
        store = populated_store(EpisodicMemoryStore())
        index = LocalRecurrenceIndex()
        index.observe_store(store)
        encoding_id = store.retrieve_by_id(
            "episode-a"
        ).evidence.encoding_provenance.encoding_id
        cross_boundary = BinarySubsequenceKey.from_window(
            encoding_id,
            torch.tensor([[1, 1], [0, 0]], dtype=torch.uint8),
        )
        self.assertEqual(index.supporters(cross_boundary), ())

    def test_repetition_in_only_one_episode_is_not_recurrent(self) -> None:
        store = EpisodicMemoryStore()
        stored = store.store(
            episode_id="episode-a",
            text="a",
            evidence=evidence(
                torch.tensor([[1, 0], [1, 0]], dtype=torch.uint8)
            ),
        )
        index = LocalRecurrenceIndex(min_length=1, max_length=1)
        index.observe(stored)
        key = BinarySubsequenceKey.from_window(
            stored.evidence.encoding_provenance.encoding_id,
            torch.tensor([[1, 0]], dtype=torch.uint8),
        )
        self.assertEqual(len(index.supporters(key)), 2)
        self.assertEqual(index.candidates(), ())

    def test_encoding_identity_prevents_cross_encoder_alias(self) -> None:
        store = EpisodicMemoryStore()
        store.store(
            episode_id="episode-a",
            text="a",
            evidence=evidence(torch.tensor([[1, 0]], dtype=torch.uint8)),
        )
        store.store(
            episode_id="episode-b",
            text="b",
            evidence=evidence(
                torch.tensor([[1, 0]], dtype=torch.uint8),
                revision="revision-b",
            ),
        )
        index = LocalRecurrenceIndex()
        index.observe_store(store)
        self.assertEqual(index.candidates(), ())
        self.assertEqual(len(index.all_keys()), 2)

    def test_restart_loaded_store_builds_identical_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_store = populated_store(PersistentEpisodicMemoryStore(root))
            first_index = LocalRecurrenceIndex()
            first_index.observe_store(first_store)

            restarted_store = PersistentEpisodicMemoryStore(root)
            restarted_index = LocalRecurrenceIndex()
            restarted_index.observe_store(restarted_store)

            first = [
                (
                    candidate.key.key_digest,
                    candidate.occurrences,
                )
                for candidate in first_index.candidates()
            ]
            restarted = [
                (
                    candidate.key.key_digest,
                    candidate.occurrences,
                )
                for candidate in restarted_index.candidates()
            ]
            self.assertEqual(restarted, first)
            self.assertEqual(
                restarted_index.indexed_window_count,
                first_index.indexed_window_count,
            )


if __name__ == "__main__":
    unittest.main()
