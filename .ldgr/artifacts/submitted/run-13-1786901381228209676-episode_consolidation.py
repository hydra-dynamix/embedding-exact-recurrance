from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

from src.episodic_encoding import SequencedBinaryAddress
from src.episodic_store import EpisodicMemoryStore, StoredEpisode


@dataclass(frozen=True)
class CanonicalEpisodeVersion:
    """One immutable version of an admitted exact whole-episode construction."""

    canonical_id: str
    version_id: str
    predecessor_version_id: str | None
    encoding_id: str
    sequenced_address: SequencedBinaryAddress
    occurrence_ids: tuple[str, ...]
    support_threshold: int
    admitted_at_stream_ordinal: int

    @property
    def support(self) -> int:
        return len(self.occurrence_ids)

    def hydrate(self, store: EpisodicMemoryStore) -> tuple[StoredEpisode, ...]:
        episodes = tuple(store.retrieve_by_id(value) for value in self.occurrence_ids)
        for episode in episodes:
            if episode.evidence.encoding_provenance.encoding_id != self.encoding_id:
                raise RuntimeError("canonical binding encoding changed")
            if episode.sequenced_address != self.sequenced_address:
                raise RuntimeError("canonical binding address changed")
        return episodes


class ExactEpisodeConsolidator:
    """Admit one canonical identity after an exact A_e recurs enough times."""

    def __init__(self, *, support_threshold: int) -> None:
        if support_threshold < 2:
            raise ValueError("support_threshold must be at least two")
        self.support_threshold = support_threshold
        self._occurrence_ids: dict[
            tuple[str, SequencedBinaryAddress],
            list[str],
        ] = {}
        self._canonical_ids: dict[
            tuple[str, SequencedBinaryAddress],
            str,
        ] = {}
        self._versions_by_canonical: dict[str, list[CanonicalEpisodeVersion]] = {}
        self._observed_episode_ids: set[str] = set()
        self._observed_episode_order: list[str] = []
        self._stream_ordinal = 0

    @staticmethod
    def _canonical_id(
        encoding_id: str,
        address: SequencedBinaryAddress,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"exact-episode-canonical-v1\0")
        digest.update(encoding_id.encode())
        digest.update(address.address_digest.encode())
        return digest.hexdigest()

    @staticmethod
    def _version_id(
        canonical_id: str,
        predecessor_version_id: str | None,
        occurrence_ids: tuple[str, ...],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"exact-episode-canonical-version-v1\0")
        digest.update(canonical_id.encode())
        digest.update((predecessor_version_id or "").encode())
        for occurrence_id in occurrence_ids:
            digest.update(occurrence_id.encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def observe(
        self,
        episode: StoredEpisode,
    ) -> CanonicalEpisodeVersion | None:
        if episode.episode_id in self._observed_episode_ids:
            raise ValueError(f"episode {episode.episode_id!r} was already observed")
        encoding_id = episode.evidence.encoding_provenance.encoding_id
        key = (encoding_id, episode.sequenced_address)
        occurrence_ids = self._occurrence_ids.setdefault(key, [])
        occurrence_ids.append(episode.episode_id)
        self._observed_episode_ids.add(episode.episode_id)
        self._observed_episode_order.append(episode.episode_id)
        ordinal = self._stream_ordinal
        self._stream_ordinal += 1
        if len(occurrence_ids) < self.support_threshold:
            return None

        canonical_id = self._canonical_ids.get(key)
        if canonical_id is None:
            canonical_id = self._canonical_id(encoding_id, episode.sequenced_address)
            self._canonical_ids[key] = canonical_id
        prior_versions = self._versions_by_canonical.setdefault(canonical_id, [])
        predecessor = prior_versions[-1].version_id if prior_versions else None
        bindings = tuple(occurrence_ids)
        version = CanonicalEpisodeVersion(
            canonical_id=canonical_id,
            version_id=self._version_id(canonical_id, predecessor, bindings),
            predecessor_version_id=predecessor,
            encoding_id=encoding_id,
            sequenced_address=episode.sequenced_address,
            occurrence_ids=bindings,
            support_threshold=self.support_threshold,
            admitted_at_stream_ordinal=ordinal,
        )
        prior_versions.append(version)
        return version

    def observe_store(
        self,
        store: EpisodicMemoryStore,
    ) -> tuple[CanonicalEpisodeVersion, ...]:
        admitted: list[CanonicalEpisodeVersion] = []
        for episode in store.episodes():
            version = self.observe(episode)
            if version is not None:
                admitted.append(version)
        return tuple(admitted)

    def canonical_for(
        self,
        episode: StoredEpisode,
    ) -> CanonicalEpisodeVersion | None:
        key = (
            episode.evidence.encoding_provenance.encoding_id,
            episode.sequenced_address,
        )
        canonical_id = self._canonical_ids.get(key)
        if canonical_id is None:
            return None
        return self._versions_by_canonical[canonical_id][-1]

    def versions(
        self,
        canonical_id: str,
    ) -> tuple[CanonicalEpisodeVersion, ...]:
        return tuple(self._versions_by_canonical.get(canonical_id, ()))

    def canonicals(self) -> tuple[CanonicalEpisodeVersion, ...]:
        return tuple(
            versions[-1]
            for _, versions in sorted(self._versions_by_canonical.items())
        )

    def save(self, root: Path) -> dict[str, object]:
        """Persist the canonical catalog without rewriting source episodes."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        canonical_rows = []
        for canonical in self.canonicals():
            versions = self.versions(canonical.canonical_id)
            canonical_rows.append(
                {
                    "canonical_id": canonical.canonical_id,
                    "encoding_id": canonical.encoding_id,
                    "address_digest": canonical.sequenced_address.address_digest,
                    "versions": [asdict(version) | {
                        "sequenced_address": {
                            "state_count": version.sequenced_address.state_count,
                            "state_dimensions": version.sequenced_address.state_dimensions,
                            "address_digest": version.sequenced_address.address_digest,
                        }
                    } for version in versions],
                }
            )
        manifest = {
            "format": "exact-episode-canonical-catalog-v1",
            "support_threshold": self.support_threshold,
            "stream_ordinal": self._stream_ordinal,
            "observed_episode_ids": self._observed_episode_order,
            "canonical_count": self.canonical_count,
            "canonicals": canonical_rows,
            "source_episodes_embedded": False,
        }
        temporary = root / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root / "manifest.json")
        return manifest

    @classmethod
    def load(
        cls,
        root: Path,
        store: EpisodicMemoryStore,
    ) -> "ExactEpisodeConsolidator":
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        if manifest["format"] != "exact-episode-canonical-catalog-v1":
            raise RuntimeError("unsupported canonical catalog format")
        consolidator = cls(support_threshold=int(manifest["support_threshold"]))
        observed = list(manifest["observed_episode_ids"])
        if len(observed) != len(set(observed)):
            raise RuntimeError("canonical catalog repeats an observed episode ID")
        for episode_id in observed:
            episode = store.retrieve_by_id(episode_id)
            key = (
                episode.evidence.encoding_provenance.encoding_id,
                episode.sequenced_address,
            )
            consolidator._occurrence_ids.setdefault(key, []).append(episode_id)
        consolidator._observed_episode_ids = set(observed)
        consolidator._observed_episode_order = observed
        consolidator._stream_ordinal = int(manifest["stream_ordinal"])
        if consolidator._stream_ordinal != len(observed):
            raise RuntimeError("canonical catalog stream ordinal changed")
        for row in manifest["canonicals"]:
            versions: list[CanonicalEpisodeVersion] = []
            for item in row["versions"]:
                occurrence_ids = tuple(item["occurrence_ids"])
                if not occurrence_ids:
                    raise RuntimeError("canonical version has no source bindings")
                source = store.retrieve_by_id(occurrence_ids[0])
                address = source.sequenced_address
                if address.address_digest != item["sequenced_address"]["address_digest"]:
                    raise RuntimeError("canonical address digest changed")
                version = CanonicalEpisodeVersion(
                    canonical_id=item["canonical_id"],
                    version_id=item["version_id"],
                    predecessor_version_id=item["predecessor_version_id"],
                    encoding_id=item["encoding_id"],
                    sequenced_address=address,
                    occurrence_ids=occurrence_ids,
                    support_threshold=int(item["support_threshold"]),
                    admitted_at_stream_ordinal=int(item["admitted_at_stream_ordinal"]),
                )
                version.hydrate(store)
                versions.append(version)
            latest = versions[-1]
            expected_id = cls._canonical_id(latest.encoding_id, latest.sequenced_address)
            if row["canonical_id"] != expected_id or latest.canonical_id != expected_id:
                raise RuntimeError("canonical identity changed")
            key = (latest.encoding_id, latest.sequenced_address)
            if tuple(consolidator._occurrence_ids[key]) != latest.occurrence_ids:
                raise RuntimeError("canonical latest bindings are incomplete")
            consolidator._canonical_ids[key] = expected_id
            consolidator._versions_by_canonical[expected_id] = versions
        if consolidator.canonical_count != int(manifest["canonical_count"]):
            raise RuntimeError("canonical catalog count changed")
        return consolidator

    @property
    def canonical_count(self) -> int:
        return len(self._canonical_ids)
