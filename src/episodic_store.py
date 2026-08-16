from dataclasses import dataclass

from src.episodic_encoding import EncodedEpisode, SequencedBinaryAddress


@dataclass(frozen=True)
class StoredEpisode:
    """One occurrence and its complete in-process encoded evidence."""

    episode_id: str
    text: str
    sequenced_address: SequencedBinaryAddress
    evidence: EncodedEpisode


class EpisodicMemoryStore:
    """Collision-safe evidence storage keyed by sequenced binary addresses A_e."""

    def __init__(self) -> None:
        self._episodes_by_sequenced_address: dict[
            SequencedBinaryAddress,
            list[StoredEpisode],
        ] = {}
        self._episodes_by_id: dict[str, StoredEpisode] = {}

    def store(
        self,
        *,
        episode_id: str,
        text: str,
        evidence: EncodedEpisode,
    ) -> StoredEpisode:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if episode_id in self._episodes_by_id:
            raise ValueError(f"episode_id {episode_id!r} is already stored")

        evidence_snapshot = EncodedEpisode(
            input_ids=evidence.input_ids.detach().cpu().clone(),
            attention_mask=evidence.attention_mask.detach().cpu().clone(),
            contextual_positions=(
                evidence.contextual_positions.detach().cpu().clone()
            ),
            raw_contextual_state_vectors=(
                evidence.raw_contextual_state_vectors.detach().cpu().clone()
            ),
            normalized_contextual_state_vectors=(
                evidence.normalized_contextual_state_vectors.detach().cpu().clone()
            ),
            binary_state_sequence=(
                evidence.binary_state_sequence.detach().cpu().clone()
            ),
        )
        sequenced_address = evidence_snapshot.sequenced_address
        episode = StoredEpisode(
            episode_id=episode_id,
            text=text,
            sequenced_address=sequenced_address,
            evidence=evidence_snapshot,
        )
        self._episodes_by_id[episode_id] = episode
        self._episodes_by_sequenced_address.setdefault(
            sequenced_address,
            [],
        ).append(episode)
        return episode

    def retrieve(
        self,
        sequenced_address: SequencedBinaryAddress,
    ) -> tuple[StoredEpisode, ...]:
        """Return every occurrence stored under the exact sequence A_e."""
        return tuple(
            self._episodes_by_sequenced_address.get(sequenced_address, ())
        )

    def retrieve_by_id(self, episode_id: str) -> StoredEpisode:
        """Hydrate one unambiguous occurrence by its episode ID."""
        try:
            return self._episodes_by_id[episode_id]
        except KeyError as error:
            raise KeyError(f"episode_id {episode_id!r} is not stored") from error

    def __len__(self) -> int:
        return len(self._episodes_by_id)
