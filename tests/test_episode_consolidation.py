from pathlib import Path
import tempfile
import unittest

import torch

from src.episode_consolidation import ExactEpisodeConsolidator
from src.episodic_encoding import EncodedEpisode, EncodingProvenance
from src.episodic_store import EpisodicMemoryStore
from src.persistent_store import PersistentEpisodicMemoryStore


def evidence(sequence: torch.Tensor, text: str) -> EncodedEpisode:
    state_count, dimensions = sequence.shape
    raw = torch.arange(state_count * dimensions, dtype=torch.float32).reshape(
        state_count,
        dimensions,
    )
    return EncodedEpisode(
        input_bytes=text.encode(),
        encoding_provenance=EncodingProvenance(
            model_id="model",
            model_revision="revision",
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


def bits(value: int) -> torch.Tensor:
    values = [(value >> shift) & 1 for shift in range(16)]
    return torch.tensor(values, dtype=torch.uint8).reshape(2, 8)


def populate(store: EpisodicMemoryStore) -> tuple[EpisodicMemoryStore, dict[str, list[str]], list[str]]:
    repeated_values = (3, 77, 991)
    repeated_ids = {f"repeat-{index}": [] for index in range(3)}
    novel_ids: list[str] = []
    for occurrence in range(5):
        for repeated_index, value in enumerate(repeated_values):
            group = f"repeat-{repeated_index}"
            episode_id = f"{group}-occurrence-{occurrence}"
            store.store(
                episode_id=episode_id,
                text=group,
                evidence=evidence(bits(value), group),
            )
            repeated_ids[group].append(episode_id)
        for novel_index in range(3):
            ordinal = occurrence * 3 + novel_index
            episode_id = f"novel-{ordinal}"
            store.store(
                episode_id=episode_id,
                text=episode_id,
                evidence=evidence(bits(2000 + ordinal), episode_id),
            )
            novel_ids.append(episode_id)
    return store, repeated_ids, novel_ids


class ExactEpisodeConsolidationTests(unittest.TestCase):
    def test_thresholds_four_and_five_create_one_canonical_per_repeated_episode(self) -> None:
        store, repeated_ids, novel_ids = populate(EpisodicMemoryStore())

        threshold_four = ExactEpisodeConsolidator(support_threshold=4)
        threshold_five = ExactEpisodeConsolidator(support_threshold=5)
        four_events = threshold_four.observe_store(store)
        five_events = threshold_five.observe_store(store)

        self.assertEqual(len(store), 30)
        self.assertEqual(threshold_four.canonical_count, 3)
        self.assertEqual(threshold_five.canonical_count, 3)
        self.assertEqual(len(four_events), 6)
        self.assertEqual(len(five_events), 3)
        for group_ids in repeated_ids.values():
            source = store.retrieve_by_id(group_ids[0])
            self.assertEqual(
                tuple(item.episode_id for item in store.retrieve(source.sequenced_address)),
                tuple(group_ids),
            )
            for consolidator in (threshold_four, threshold_five):
                canonical = consolidator.canonical_for(source)
                self.assertIsNotNone(canonical)
                assert canonical is not None
                self.assertEqual(canonical.occurrence_ids, tuple(group_ids))
                self.assertEqual(canonical.support, 5)
                self.assertEqual(
                    tuple(item.episode_id for item in canonical.hydrate(store)),
                    tuple(group_ids),
                )
                for occurrence_id in group_ids:
                    self.assertEqual(
                        store.retrieve_by_id(occurrence_id).sequenced_address,
                        canonical.sequenced_address,
                    )
        for episode_id in novel_ids:
            episode = store.retrieve_by_id(episode_id)
            self.assertEqual(len(store.retrieve(episode.sequenced_address)), 1)
            self.assertIsNone(threshold_four.canonical_for(episode))
            self.assertIsNone(threshold_five.canonical_for(episode))

    def test_threshold_four_versions_preserve_admission_and_later_binding(self) -> None:
        store, repeated_ids, _ = populate(EpisodicMemoryStore())
        consolidator = ExactEpisodeConsolidator(support_threshold=4)
        consolidator.observe_store(store)

        for group_ids in repeated_ids.values():
            canonical = consolidator.canonical_for(store.retrieve_by_id(group_ids[0]))
            assert canonical is not None
            versions = consolidator.versions(canonical.canonical_id)
            self.assertEqual([item.support for item in versions], [4, 5])
            self.assertIsNone(versions[0].predecessor_version_id)
            self.assertEqual(
                versions[1].predecessor_version_id,
                versions[0].version_id,
            )
            self.assertEqual(versions[0].occurrence_ids, tuple(group_ids[:4]))
            self.assertEqual(versions[1].occurrence_ids, tuple(group_ids))

    def test_restart_loads_persisted_identical_canonical_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _, _ = populate(PersistentEpisodicMemoryStore(root / "evidence"))
            first_four = ExactEpisodeConsolidator(support_threshold=4)
            first_four.observe_store(first)
            first_four.save(root / "canonicals-4")

            restarted = PersistentEpisodicMemoryStore(root / "evidence")
            restarted_four = ExactEpisodeConsolidator.load(
                root / "canonicals-4",
                restarted,
            )

            self.assertEqual(
                [
                    (item.canonical_id, item.version_id, item.occurrence_ids)
                    for item in restarted_four.canonicals()
                ],
                [
                    (item.canonical_id, item.version_id, item.occurrence_ids)
                    for item in first_four.canonicals()
                ],
            )


if __name__ == "__main__":
    unittest.main()
