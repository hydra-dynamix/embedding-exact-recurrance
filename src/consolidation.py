from dataclasses import dataclass
import hashlib

import torch

from src.episodic_store import EpisodicMemoryStore
from src.recurrence import (
    BinarySubsequenceKey,
    OccurrenceSpan,
    RecurrenceCandidate,
    hydrate_occurrence,
)


@dataclass(frozen=True)
class ResidualBinding:
    """Occurrence-specific B components outside a shared construction C."""

    occurrence: OccurrenceSpan
    state_dimensions: int
    residual_values: bytes

    def __post_init__(self) -> None:
        expected = self.occurrence.length * self.state_dimensions
        if self.state_dimensions <= 0:
            raise ValueError("state_dimensions must be positive")
        if len(self.residual_values) != expected:
            raise ValueError("residual_values cardinality does not match occurrence")
        if any(value not in (0, 1) for value in self.residual_values):
            raise ValueError("residual_values must contain only zero and one")

    @classmethod
    def from_tensor(
        cls,
        occurrence: OccurrenceSpan,
        residual: torch.Tensor,
    ) -> "ResidualBinding":
        if residual.ndim != 2:
            raise ValueError("residual must have shape [length,D]")
        if residual.shape[0] != occurrence.length:
            raise ValueError("residual length does not match occurrence")
        if not torch.all((residual == 0) | (residual == 1)):
            raise ValueError("residual must contain only zero and one")
        values = bytes(
            residual.detach()
            .cpu()
            .to(torch.uint8)
            .contiguous()
            .flatten()
            .tolist()
        )
        return cls(
            occurrence=occurrence,
            state_dimensions=int(residual.shape[1]),
            residual_values=values,
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            tuple(self.residual_values),
            dtype=torch.uint8,
        ).reshape(self.occurrence.length, self.state_dimensions)


@dataclass(frozen=True)
class ConstructionCandidate:
    """Lossless shared C plus complete residual bindings; never auto-admitted."""

    family_id: str
    version_id: str
    shared_component: BinarySubsequenceKey
    bindings: tuple[ResidualBinding, ...]
    source_recurrence_key_digests: tuple[str, ...]
    predecessor_version_id: str | None = None
    admitted: bool = False

    def __post_init__(self) -> None:
        if not self.family_id or not self.version_id:
            raise ValueError("construction IDs must not be empty")
        if self.admitted:
            raise ValueError("candidate construction cannot be auto-admitted")
        if len(self.bindings) < 2:
            raise ValueError("construction candidate needs at least two bindings")
        occurrences = [binding.occurrence for binding in self.bindings]
        if len(set(occurrences)) != len(occurrences):
            raise ValueError("construction bindings contain duplicate occurrences")
        if len({item.episode_id for item in occurrences}) < 2:
            raise ValueError("construction needs support from distinct episodes")
        for binding in self.bindings:
            occurrence = binding.occurrence
            if occurrence.encoding_id != self.shared_component.encoding_id:
                raise ValueError("binding encoding differs from construction")
            if occurrence.length != self.shared_component.length:
                raise ValueError("binding length differs from construction")
            if binding.state_dimensions != self.shared_component.state_dimensions:
                raise ValueError("binding dimensions differ from construction")

    @property
    def occurrences(self) -> tuple[OccurrenceSpan, ...]:
        return tuple(binding.occurrence for binding in self.bindings)

    def reconstruct(self, occurrence: OccurrenceSpan) -> torch.Tensor:
        """Exactly reconstruct one source B window as C union its residual."""
        for binding in self.bindings:
            if binding.occurrence == occurrence:
                return torch.bitwise_or(
                    self.shared_component.to_binary_window(),
                    binding.to_tensor(),
                )
        raise KeyError("occurrence is not bound to this construction")


class ConstructionCatalog:
    """Append-only in-process catalog of immutable candidate versions."""

    def __init__(self) -> None:
        self._by_version: dict[str, ConstructionCandidate] = {}
        self._versions_by_family: dict[str, list[str]] = {}

    def add(self, candidate: ConstructionCandidate) -> None:
        if candidate.version_id in self._by_version:
            raise ValueError("construction version already exists")
        self._by_version[candidate.version_id] = candidate
        self._versions_by_family.setdefault(candidate.family_id, []).append(
            candidate.version_id
        )

    def version(self, version_id: str) -> ConstructionCandidate:
        return self._by_version[version_id]

    def family_versions(
        self,
        family_id: str,
    ) -> tuple[ConstructionCandidate, ...]:
        return tuple(
            self._by_version[version_id]
            for version_id in self._versions_by_family.get(family_id, ())
        )


def _construction_ids(
    shared: BinarySubsequenceKey,
    bindings: tuple[ResidualBinding, ...],
    predecessor_version_id: str | None,
    family_id: str | None,
) -> tuple[str, str]:
    if family_id is None:
        family_digest = hashlib.sha256()
        family_digest.update(b"binary-construction-family-v1\0")
        family_digest.update(shared.encoding_id.encode())
        family_digest.update(shared.length.to_bytes(8, "big"))
        family_digest.update(shared.state_dimensions.to_bytes(8, "big"))
        family_digest.update(shared.binary_state_values)
        family_id = family_digest.hexdigest()

    version_digest = hashlib.sha256()
    version_digest.update(b"binary-construction-version-v1\0")
    version_digest.update(family_id.encode())
    version_digest.update((predecessor_version_id or "").encode())
    version_digest.update(shared.key_digest.encode())
    for binding in bindings:
        occurrence = binding.occurrence
        version_digest.update(occurrence.episode_id.encode())
        version_digest.update(occurrence.start_offset.to_bytes(8, "big"))
        version_digest.update(occurrence.length.to_bytes(8, "big"))
        version_digest.update(occurrence.encoding_id.encode())
        version_digest.update(binding.residual_values)
    return family_id, version_digest.hexdigest()


def factor_occurrences(
    store: EpisodicMemoryStore,
    occurrences: tuple[OccurrenceSpan, ...],
    *,
    source_recurrence_key_digests: tuple[str, ...] = (),
    predecessor_version_id: str | None = None,
    construction_family_id: str | None = None,
) -> ConstructionCandidate:
    """Factor explicitly bound same-shape windows into C and exact residuals."""
    if len(occurrences) < 2:
        raise ValueError("at least two occurrences are required")
    if len(set(occurrences)) != len(occurrences):
        raise ValueError("occurrences must be unique")
    if len({item.episode_id for item in occurrences}) < 2:
        raise ValueError("occurrences must span at least two episodes")
    ordered = tuple(sorted(occurrences))
    windows = [hydrate_occurrence(store, occurrence) for occurrence in ordered]
    first_shape = windows[0].shape
    encoding_id = ordered[0].encoding_id
    for occurrence, window in zip(ordered, windows, strict=True):
        if occurrence.encoding_id != encoding_id:
            raise ValueError("occurrences use different encodings")
        if window.shape != first_shape:
            raise ValueError("occurrence windows have different shapes")
    shared_tensor = windows[0].clone()
    for window in windows[1:]:
        shared_tensor = torch.bitwise_and(shared_tensor, window)
    shared = BinarySubsequenceKey.from_window(encoding_id, shared_tensor)
    bindings: list[ResidualBinding] = []
    for occurrence, window in zip(ordered, windows, strict=True):
        residual = torch.bitwise_and(window, 1 - shared_tensor)
        binding = ResidualBinding.from_tensor(occurrence, residual)
        reconstructed = torch.bitwise_or(shared_tensor, binding.to_tensor())
        if not torch.equal(reconstructed, window):
            raise RuntimeError("construction residual failed exact reconstruction")
        bindings.append(binding)
    binding_tuple = tuple(bindings)
    family_id, version_id = _construction_ids(
        shared,
        binding_tuple,
        predecessor_version_id,
        construction_family_id,
    )
    return ConstructionCandidate(
        family_id=family_id,
        version_id=version_id,
        shared_component=shared,
        bindings=binding_tuple,
        source_recurrence_key_digests=tuple(
            sorted(set(source_recurrence_key_digests))
        ),
        predecessor_version_id=predecessor_version_id,
        admitted=False,
    )


def factor_recurrence_candidate(
    store: EpisodicMemoryStore,
    recurrence: RecurrenceCandidate,
) -> ConstructionCandidate:
    return factor_occurrences(
        store,
        recurrence.occurrences,
        source_recurrence_key_digests=(recurrence.key.key_digest,),
    )


def revise_construction(
    store: EpisodicMemoryStore,
    predecessor: ConstructionCandidate,
    additional_occurrences: tuple[OccurrenceSpan, ...],
) -> ConstructionCandidate:
    """Create a new immutable version while retaining the predecessor."""
    return factor_occurrences(
        store,
        predecessor.occurrences + additional_occurrences,
        source_recurrence_key_digests=(
            *predecessor.source_recurrence_key_digests,
        ),
        predecessor_version_id=predecessor.version_id,
        construction_family_id=predecessor.family_id,
    )
