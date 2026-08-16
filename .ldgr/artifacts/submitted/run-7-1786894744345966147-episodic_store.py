from dataclasses import dataclass

from src.episodic_encoding import EpisodeAddress


@dataclass(frozen=True)
class StoredEpisode:
    episode_id: str
    text: str
    address: EpisodeAddress


class EpisodicMemoryStore:
    """Collision-safe exact storage indexed by binary episode addresses."""

    def __init__(self) -> None:
        self._episodes_by_address: dict[EpisodeAddress, list[StoredEpisode]] = {}
        self._episodes_by_id: dict[str, StoredEpisode] = {}

    def store(
        self,
        *,
        episode_id: str,
        text: str,
        address: EpisodeAddress,
    ) -> StoredEpisode:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if episode_id in self._episodes_by_id:
            raise ValueError(f"episode_id {episode_id!r} is already stored")

        episode = StoredEpisode(
            episode_id=episode_id,
            text=text,
            address=address,
        )
        self._episodes_by_id[episode_id] = episode
        self._episodes_by_address.setdefault(address, []).append(episode)
        return episode

    def retrieve(self, address: EpisodeAddress) -> tuple[StoredEpisode, ...]:
        """Return every episode at an exact address without hiding collisions."""
        return tuple(self._episodes_by_address.get(address, ()))

    def retrieve_by_id(self, episode_id: str) -> StoredEpisode:
        try:
            return self._episodes_by_id[episode_id]
        except KeyError as error:
            raise KeyError(f"episode_id {episode_id!r} is not stored") from error

    def __len__(self) -> int:
        return len(self._episodes_by_id)
