from dataclasses import dataclass
import hashlib

import torch

from src.episodic_store import EpisodicMemoryStore, StoredEpisode


@dataclass(frozen=True)
class BinarySubsequenceKey:
    """Exact ordered local sequence (B_i, ..., B_(i+L-1))."""

    encoding_id: str
    length: int
    state_dimensions: int
    binary_state_values: bytes

    def __post_init__(self) -> None:
        if not self.encoding_id:
            raise ValueError("encoding_id must not be empty")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.state_dimensions <= 0:
            raise ValueError("state_dimensions must be positive")
        if len(self.binary_state_values) != self.length * self.state_dimensions:
            raise ValueError("binary_state_values cardinality does not match shape")
        if any(value not in (0, 1) for value in self.binary_state_values):
            raise ValueError("binary_state_values must contain only zero and one")

    @classmethod
    def from_window(
        cls,
        encoding_id: str,
        binary_window: torch.Tensor,
    ) -> "BinarySubsequenceKey":
        if binary_window.ndim != 2 or binary_window.numel() == 0:
            raise ValueError("binary_window must have nonempty shape [length,D]")
        if not torch.all((binary_window == 0) | (binary_window == 1)):
            raise ValueError("binary_window must contain only zero and one")
        values = bytes(
            binary_window.detach()
            .cpu()
            .to(torch.uint8)
            .contiguous()
            .flatten()
            .tolist()
        )
        return cls(
            encoding_id=encoding_id,
            length=int(binary_window.shape[0]),
            state_dimensions=int(binary_window.shape[1]),
            binary_state_values=values,
        )

    @property
    def key_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.encoding_id.encode())
        digest.update(self.length.to_bytes(8, "big"))
        digest.update(self.state_dimensions.to_bytes(8, "big"))
        digest.update(self.binary_state_values)
        return digest.hexdigest()

    def to_binary_window(self) -> torch.Tensor:
        return torch.tensor(
            tuple(self.binary_state_values),
            dtype=torch.uint8,
        ).reshape(self.length, self.state_dimensions)


@dataclass(frozen=True, order=True)
class OccurrenceSpan:
    """Stable source binding for one local window inside one episode."""

    episode_id: str
    start_offset: int
    length: int
    encoding_id: str

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.start_offset < 0:
            raise ValueError("start_offset must be nonnegative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not self.encoding_id:
            raise ValueError("encoding_id must not be empty")


@dataclass(frozen=True)
class RecurrenceCandidate:
    """Complete supporters for an exact local key; not an admitted construction."""

    key: BinarySubsequenceKey
    occurrences: tuple[OccurrenceSpan, ...]

    @property
    def distinct_episode_count(self) -> int:
        return len({occurrence.episode_id for occurrence in self.occurrences})


class LocalRecurrenceIndex:
    """Complete uncapped exact recurrence over all legal local B windows."""

    def __init__(
        self,
        *,
        min_length: int = 1,
        max_length: int | None = None,
        minimum_distinct_episodes: int = 2,
    ) -> None:
        if min_length <= 0:
            raise ValueError("min_length must be positive")
        if max_length is not None and max_length < min_length:
            raise ValueError("max_length must be at least min_length")
        if minimum_distinct_episodes < 2:
            raise ValueError("minimum_distinct_episodes must be at least two")
        self.min_length = min_length
        self.max_length = max_length
        self.minimum_distinct_episodes = minimum_distinct_episodes
        self._supporters: dict[BinarySubsequenceKey, list[OccurrenceSpan]] = {}
        self._observed_episode_ids: set[str] = set()
        self._indexed_window_count = 0

    def observe(self, episode: StoredEpisode) -> int:
        if episode.episode_id in self._observed_episode_ids:
            raise ValueError(f"episode {episode.episode_id!r} was already observed")
        sequence = episode.evidence.binary_state_sequence.detach().cpu()
        if sequence.ndim != 2 or sequence.numel() == 0:
            raise ValueError("episode binary_state_sequence must have shape [T,D]")
        encoding_id = episode.evidence.encoding_provenance.encoding_id
        maximum = sequence.shape[0]
        if self.max_length is not None:
            maximum = min(maximum, self.max_length)
        added = 0
        for length in range(self.min_length, maximum + 1):
            for start in range(0, sequence.shape[0] - length + 1):
                key = BinarySubsequenceKey.from_window(
                    encoding_id,
                    sequence[start : start + length],
                )
                occurrence = OccurrenceSpan(
                    episode_id=episode.episode_id,
                    start_offset=start,
                    length=length,
                    encoding_id=encoding_id,
                )
                self._supporters.setdefault(key, []).append(occurrence)
                added += 1
        self._observed_episode_ids.add(episode.episode_id)
        self._indexed_window_count += added
        return added

    def observe_store(self, store: EpisodicMemoryStore) -> int:
        return sum(self.observe(episode) for episode in store.episodes())

    def supporters(
        self,
        key: BinarySubsequenceKey,
    ) -> tuple[OccurrenceSpan, ...]:
        """Return every supporting occurrence without ranking or a result cap."""
        return tuple(self._supporters.get(key, ()))

    def candidates(self) -> tuple[RecurrenceCandidate, ...]:
        result = [
            RecurrenceCandidate(key=key, occurrences=tuple(occurrences))
            for key, occurrences in self._supporters.items()
            if len({item.episode_id for item in occurrences})
            >= self.minimum_distinct_episodes
        ]
        return tuple(sorted(result, key=lambda item: item.key.key_digest))

    def all_keys(self) -> tuple[BinarySubsequenceKey, ...]:
        return tuple(sorted(self._supporters, key=lambda key: key.key_digest))

    @property
    def indexed_window_count(self) -> int:
        return self._indexed_window_count


def hydrate_occurrence(
    store: EpisodicMemoryStore,
    occurrence: OccurrenceSpan,
) -> torch.Tensor:
    """Hydrate the exact B window referenced by an occurrence span."""
    episode = store.retrieve_by_id(occurrence.episode_id)
    evidence = episode.evidence
    if evidence.encoding_provenance.encoding_id != occurrence.encoding_id:
        raise RuntimeError("occurrence encoding identity changed")
    stop = occurrence.start_offset + occurrence.length
    sequence = evidence.binary_state_sequence
    if stop > sequence.shape[0]:
        raise RuntimeError("occurrence extends beyond its source episode")
    return sequence[occurrence.start_offset:stop].detach().cpu().clone()
