from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import torch

from src.episodic_encoding import (
    EncodedEpisode,
    EncodingProvenance,
    SequencedBinaryAddress,
)
from src.episodic_store import EpisodicMemoryStore, StoredEpisode


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_tensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "compression": "none",
    }


def load_tensor(root: Path, artifact: dict[str, object]) -> torch.Tensor:
    path = root / str(artifact["path"])
    if path.stat().st_size != artifact["bytes"]:
        raise RuntimeError(f"tensor byte count changed: {path}")
    if file_sha256(path) != artifact["sha256"]:
        raise RuntimeError(f"tensor digest changed: {path}")
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if list(array.shape) != artifact["shape"]:
        raise RuntimeError(f"tensor shape changed: {path}")
    tensor = torch.from_numpy(array.copy())
    if str(tensor.dtype) != artifact["dtype"]:
        raise RuntimeError(f"tensor dtype changed: {path}")
    return tensor


class PersistentEpisodicMemoryStore(EpisodicMemoryStore):
    """Uncompressed restartable evidence storage physically rooted under A_e."""

    FORMAT_VERSION = "sequenced-binary-evidence-v1"

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.address_root = self.root / "addresses"
        self.episode_index_root = self.root / "episode-index"
        self.address_root.mkdir(parents=True, exist_ok=True)
        self.episode_index_root.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    @staticmethod
    def _episode_key(episode_id: str) -> str:
        return hashlib.sha256(episode_id.encode()).hexdigest()

    def _address_directory(self, address: SequencedBinaryAddress) -> Path:
        return self.address_root / address.address_digest

    def _ensure_address(self, address: SequencedBinaryAddress) -> Path:
        directory = self._address_directory(address)
        manifest_path = directory / "address.json"
        values_path = directory / "binary-state-values.u8.raw"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest != {
                "address_digest": address.address_digest,
                "binary_values": {
                    "bytes": len(address.binary_state_values),
                    "path": values_path.name,
                    "sha256": hashlib.sha256(address.binary_state_values).hexdigest(),
                },
                "format": self.FORMAT_VERSION,
                "state_count": address.state_count,
                "state_dimensions": address.state_dimensions,
            }:
                raise RuntimeError("sequenced address manifest mismatch")
            if values_path.read_bytes() != address.binary_state_values:
                raise RuntimeError("sequenced address bytes mismatch")
            return directory
        directory.mkdir(parents=True, exist_ok=True)
        values_path.write_bytes(address.binary_state_values)
        write_json_atomic(
            manifest_path,
            {
                "address_digest": address.address_digest,
                "binary_values": {
                    "bytes": len(address.binary_state_values),
                    "path": values_path.name,
                    "sha256": file_sha256(values_path),
                },
                "format": self.FORMAT_VERSION,
                "state_count": address.state_count,
                "state_dimensions": address.state_dimensions,
            },
        )
        return directory

    def _persist_episode(
        self,
        episode_id: str,
        text: str,
        evidence: EncodedEpisode,
        store_ordinal: int,
    ) -> None:
        address = evidence.sequenced_address
        address_directory = self._ensure_address(address)
        episode_key = self._episode_key(episode_id)
        episode_directory = address_directory / "episodes" / episode_key
        if episode_directory.exists():
            raise ValueError(f"episode_id {episode_id!r} is already persisted")
        temporary = episode_directory.with_name(
            episode_directory.name + f".tmp-{uuid4().hex}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            input_path = temporary / "input.bin"
            input_path.write_bytes(evidence.input_bytes)
            tensors = {
                "input_ids": save_tensor(temporary / "input_ids.npy", evidence.input_ids),
                "attention_mask": save_tensor(temporary / "attention_mask.npy", evidence.attention_mask),
                "contextual_positions": save_tensor(temporary / "contextual_positions.npy", evidence.contextual_positions),
                "H": save_tensor(temporary / "H.npy", evidence.raw_contextual_state_vectors),
                "N": save_tensor(temporary / "N.npy", evidence.normalized_contextual_state_vectors),
                "binary_state_sequence": save_tensor(temporary / "B.npy", evidence.binary_state_sequence),
            }
            metadata = {
                "address_digest": address.address_digest,
                "encoding_id": evidence.encoding_provenance.encoding_id,
                "encoding_provenance": asdict(evidence.encoding_provenance),
                "episode_id": episode_id,
                "format": self.FORMAT_VERSION,
                "input": {
                    "bytes": input_path.stat().st_size,
                    "path": input_path.name,
                    "sha256": file_sha256(input_path),
                },
                "store_ordinal": store_ordinal,
                "sequenced_address": {
                    "state_count": address.state_count,
                    "state_dimensions": address.state_dimensions,
                },
                "tensors": tensors,
                "text": text,
            }
            write_json_atomic(temporary / "metadata.json", metadata)
            episode_directory.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, episode_directory)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        write_json_atomic(
            self.episode_index_root / f"{episode_key}.json",
            {
                "address_digest": address.address_digest,
                "episode_id": episode_id,
                "metadata": str(
                    episode_directory.relative_to(self.root) / "metadata.json"
                ),
            },
        )

    def store(
        self,
        *,
        episode_id: str,
        text: str,
        evidence: EncodedEpisode,
    ) -> StoredEpisode:
        if episode_id in self._episodes_by_id:
            raise ValueError(f"episode_id {episode_id!r} is already stored")
        self._persist_episode(
            episode_id,
            text,
            evidence,
            store_ordinal=len(self),
        )
        return super().store(
            episode_id=episode_id,
            text=text,
            evidence=evidence,
        )

    def _load_episode(
        self,
        metadata_path: Path,
    ) -> tuple[int, str, str, EncodedEpisode]:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["format"] != self.FORMAT_VERSION:
            raise RuntimeError("unsupported persistent episode format")
        input_artifact = metadata["input"]
        input_path = metadata_path.parent / input_artifact["path"]
        if input_path.stat().st_size != input_artifact["bytes"]:
            raise RuntimeError("input byte count changed")
        if file_sha256(input_path) != input_artifact["sha256"]:
            raise RuntimeError("input digest changed")
        provenance = EncodingProvenance(**metadata["encoding_provenance"])
        if provenance.encoding_id != metadata["encoding_id"]:
            raise RuntimeError("encoding provenance digest changed")
        tensors = {
            name: load_tensor(metadata_path.parent, artifact)
            for name, artifact in metadata["tensors"].items()
        }
        evidence = EncodedEpisode(
            input_bytes=input_path.read_bytes(),
            encoding_provenance=provenance,
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            contextual_positions=tensors["contextual_positions"],
            raw_contextual_state_vectors=tensors["H"],
            normalized_contextual_state_vectors=tensors["N"],
            binary_state_sequence=tensors["binary_state_sequence"],
        )
        if evidence.sequenced_address.address_digest != metadata["address_digest"]:
            raise RuntimeError("hydrated sequenced address changed")
        return (
            int(metadata["store_ordinal"]),
            metadata["episode_id"],
            metadata["text"],
            evidence,
        )

    def _load_existing(self) -> None:
        hydrated: list[
            tuple[int, str, str, EncodedEpisode, dict[str, object]]
        ] = []
        for pointer_path in self.episode_index_root.glob("*.json"):
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            metadata_path = self.root / pointer["metadata"]
            ordinal, episode_id, text, evidence = self._load_episode(metadata_path)
            hydrated.append((ordinal, episode_id, text, evidence, pointer))
        ordinals = [item[0] for item in hydrated]
        if len(set(ordinals)) != len(ordinals):
            raise RuntimeError("duplicate persistent store ordinal")
        for expected_ordinal, item in enumerate(sorted(hydrated)):
            ordinal, episode_id, text, evidence, pointer = item
            if ordinal != expected_ordinal:
                raise RuntimeError("persistent store ordinal gap")
            if episode_id != pointer["episode_id"]:
                raise RuntimeError("episode pointer ID mismatch")
            if evidence.sequenced_address.address_digest != pointer["address_digest"]:
                raise RuntimeError("episode pointer address mismatch")
            super().store(
                episode_id=episode_id,
                text=text,
                evidence=evidence,
            )
