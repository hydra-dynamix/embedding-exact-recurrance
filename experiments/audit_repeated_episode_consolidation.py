from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.episode_consolidation import ExactEpisodeConsolidator
from src.persistent_store import PersistentEpisodicMemoryStore


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def audit(root: Path) -> dict[str, object]:
    results = json.loads((root / "results.json").read_text())
    evidence_root = root / "evidence"
    before = directory_digest(evidence_root)
    store = PersistentEpisodicMemoryStore(evidence_root)
    consolidators = {
        threshold: ExactEpisodeConsolidator.load(
            root / f"canonicals-support-{threshold}",
            store,
        )
        for threshold in (4, 5)
    }
    repeated_complete = True
    canonical_complete = True
    for group in range(3):
        occurrence_ids = tuple(
            f"repeat-{group}-occurrence-{occurrence}" for occurrence in range(5)
        )
        source = store.retrieve_by_id(occurrence_ids[0])
        repeated_complete &= tuple(
            item.episode_id for item in store.retrieve(source.sequenced_address)
        ) == occurrence_ids
        for consolidator in consolidators.values():
            canonical = consolidator.canonical_for(source)
            canonical_complete &= canonical is not None
            if canonical is not None:
                canonical_complete &= canonical.occurrence_ids == occurrence_ids
                canonical_complete &= all(
                    episode.sequenced_address == canonical.sequenced_address
                    for episode in canonical.hydrate(store)
                )
    novel_singletons = True
    for novel in range(15):
        episode = store.retrieve_by_id(f"novel-{novel}")
        novel_singletons &= len(store.retrieve(episode.sequenced_address)) == 1
        novel_singletons &= all(
            consolidator.canonical_for(episode) is None
            for consolidator in consolidators.values()
        )
    after = directory_digest(evidence_root)
    checks = {
        "treatment_reported_pass": results["pass"] is True,
        "all_treatment_checks_passed": all(results["checks"].values()),
        "all_30_occurrences_restart_hydrate": len(store) == 30,
        "all_three_repeated_buckets_return_five_occurrences": repeated_complete,
        "both_thresholds_load_three_singular_canonicals": all(
            consolidator.canonical_count == 3
            for consolidator in consolidators.values()
        ),
        "all_canonical_bindings_hydrate_and_replay": canonical_complete,
        "all_15_novel_episodes_are_unconsolidated_singletons": novel_singletons,
        "threshold_4_versions_are_support_4_then_5": all(
            [version.support for version in consolidators[4].versions(canonical.canonical_id)]
            == [4, 5]
            for canonical in consolidators[4].canonicals()
        ),
        "threshold_5_versions_are_support_5": all(
            [version.support for version in consolidators[5].versions(canonical.canonical_id)]
            == [5]
            for canonical in consolidators[5].canonicals()
        ),
        "source_evidence_unchanged_during_query": before == after,
        "source_episodes_not_embedded_in_canonical_catalog": all(
            json.loads(
                (root / f"canonicals-support-{threshold}" / "manifest.json").read_text()
            )["source_episodes_embedded"]
            is False
            for threshold in (4, 5)
        ),
    }
    result = {
        "experiment_id": results["experiment_id"],
        "checks": checks,
        "pass": all(checks.values()),
        "audited_occurrences": len(store),
        "audited_canonicals_per_threshold": 3,
        "audited_thresholds": [4, 5],
        "evidence_directory_sha256": after,
    }
    (root / "independent-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.root)
    print(json.dumps({"experiment_id": result["experiment_id"], "pass": result["pass"]}))
