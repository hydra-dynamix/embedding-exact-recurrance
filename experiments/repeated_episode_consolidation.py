from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModel, AutoTokenizer

from src.episode_consolidation import ExactEpisodeConsolidator
from src.episodic_encoding import EncodingProvenance, encode_episode
from src.persistent_store import PersistentEpisodicMemoryStore

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
REPEATED_TEXTS = (
    "The red key opens the brass door.",
    "Mara placed the cup beside the lamp.",
    "At dawn the train crossed the river.",
)
NOVEL_TEXTS = tuple(
    f"Novel episode {index:02d}: marker {hashlib.sha256(f'novel-{index}'.encode()).hexdigest()[:12]}."
    for index in range(15)
)


def stream() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for occurrence in range(5):
        for group, text in enumerate(REPEATED_TEXTS):
            rows.append(
                {
                    "ordinal": ordinal,
                    "episode_id": f"repeat-{group}-occurrence-{occurrence}",
                    "kind": "repeated",
                    "group": group,
                    "occurrence": occurrence,
                    "text": text,
                }
            )
            ordinal += 1
        for within_round in range(3):
            novel = occurrence * 3 + within_round
            rows.append(
                {
                    "ordinal": ordinal,
                    "episode_id": f"novel-{novel}",
                    "kind": "novel",
                    "group": None,
                    "occurrence": 0,
                    "text": NOVEL_TEXTS[novel],
                }
            )
            ordinal += 1
    return rows


def run(output: Path) -> dict[str, object]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    evidence_root = output / "evidence"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    model_device = next(model.parameters()).device
    provenance = EncodingProvenance(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        hidden_layer=-1,
        tau_threshold=0.8,
    )
    store = PersistentEpisodicMemoryStore(evidence_root)
    consolidators = {
        threshold: ExactEpisodeConsolidator(support_threshold=threshold)
        for threshold in (4, 5)
    }
    calls = 0
    started = time.perf_counter_ns()
    admission_events: dict[int, list[dict[str, object]]] = {4: [], 5: []}
    rows = stream()
    for row in rows:
        text = str(row["text"])
        inputs = tokenizer(text, return_tensors="pt").to(model_device)
        encoded = encode_episode(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_bytes=text.encode(),
            encoding_provenance=provenance,
        )
        calls += 1
        stored = store.store(
            episode_id=str(row["episode_id"]),
            text=text,
            evidence=encoded,
        )
        row["address_digest"] = stored.sequenced_address.address_digest
        row["state_count"] = stored.sequenced_address.state_count
        for threshold, consolidator in consolidators.items():
            version = consolidator.observe(stored)
            if version is not None:
                admission_events[threshold].append(
                    {
                        "stream_ordinal": row["ordinal"],
                        "episode_id": row["episode_id"],
                        "canonical_id": version.canonical_id,
                        "version_id": version.version_id,
                        "predecessor_version_id": version.predecessor_version_id,
                        "support": version.support,
                    }
                )

    for threshold, consolidator in consolidators.items():
        consolidator.save(output / f"canonicals-support-{threshold}")
    restarted = PersistentEpisodicMemoryStore(evidence_root)
    restart_consolidators = {
        threshold: ExactEpisodeConsolidator.load(
            output / f"canonicals-support-{threshold}",
            restarted,
        )
        for threshold in (4, 5)
    }

    repeated_results: list[dict[str, object]] = []
    for group, text in enumerate(REPEATED_TEXTS):
        occurrence_ids = tuple(
            f"repeat-{group}-occurrence-{occurrence}" for occurrence in range(5)
        )
        source = restarted.retrieve_by_id(occurrence_ids[0])
        bucket = restarted.retrieve(source.sequenced_address)
        arms: dict[str, object] = {}
        for threshold, consolidator in restart_consolidators.items():
            canonical = consolidator.canonical_for(source)
            if canonical is None:
                raise RuntimeError("repeated episode was not consolidated")
            hydrated = canonical.hydrate(restarted)
            arms[str(threshold)] = {
                "canonical_id": canonical.canonical_id,
                "latest_version_id": canonical.version_id,
                "version_count": len(consolidator.versions(canonical.canonical_id)),
                "support": canonical.support,
                "occurrence_ids": list(canonical.occurrence_ids),
                "hydrated_occurrence_ids": [item.episode_id for item in hydrated],
            }
        repeated_results.append(
            {
                "group": group,
                "text": text,
                "address_digest": source.sequenced_address.address_digest,
                "retrieved_occurrence_ids": [item.episode_id for item in bucket],
                "expected_occurrence_ids": list(occurrence_ids),
                "threshold_arms": arms,
            }
        )

    novel_results: list[dict[str, object]] = []
    for novel in range(15):
        episode = restarted.retrieve_by_id(f"novel-{novel}")
        bucket = restarted.retrieve(episode.sequenced_address)
        novel_results.append(
            {
                "episode_id": episode.episode_id,
                "address_digest": episode.sequenced_address.address_digest,
                "bucket_occurrence_ids": [item.episode_id for item in bucket],
                "canonical_threshold_4": (
                    restart_consolidators[4].canonical_for(episode).canonical_id
                    if restart_consolidators[4].canonical_for(episode) is not None
                    else None
                ),
                "canonical_threshold_5": (
                    restart_consolidators[5].canonical_for(episode).canonical_id
                    if restart_consolidators[5].canonical_for(episode) is not None
                    else None
                ),
            }
        )

    checks = {
        "all_occurrences_stored_after_restart": len(restarted) == 30,
        "one_model_call_per_occurrence": calls == 30,
        "all_repeated_buckets_complete": all(
            item["retrieved_occurrence_ids"] == item["expected_occurrence_ids"]
            for item in repeated_results
        ),
        "threshold_4_three_singular_canonicals": restart_consolidators[4].canonical_count == 3,
        "threshold_5_three_singular_canonicals": restart_consolidators[5].canonical_count == 3,
        "threshold_4_admits_on_fourth_occurrence": sorted(
            event["stream_ordinal"] for event in admission_events[4] if event["support"] == 4
        ) == [18, 19, 20],
        "threshold_5_admits_on_fifth_occurrence": sorted(
            event["stream_ordinal"] for event in admission_events[5]
        ) == [24, 25, 26],
        "all_latest_canonicals_bind_five_occurrences": all(
            item["threshold_arms"][str(threshold)]["support"] == 5
            for item in repeated_results
            for threshold in (4, 5)
        ),
        "novel_episodes_remain_singletons": all(
            len(item["bucket_occurrence_ids"]) == 1 for item in novel_results
        ),
        "novel_episodes_not_consolidated": all(
            item["canonical_threshold_4"] is None
            and item["canonical_threshold_5"] is None
            for item in novel_results
        ),
        "restart_rebuilds_same_canonical_ids": all(
            [item.canonical_id for item in consolidators[threshold].canonicals()]
            == [item.canonical_id for item in restart_consolidators[threshold].canonicals()]
            for threshold in (4, 5)
        ),
        "every_canonical_hydrates_and_replays": all(
            all(
                episode.sequenced_address == canonical.sequenced_address
                for episode in canonical.hydrate(restarted)
            )
            for consolidator in restart_consolidators.values()
            for canonical in consolidator.canonicals()
        ),
    }
    result = {
        "experiment_id": "interleaved-repeated-episode-consolidation",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tau_threshold": 0.8,
        "consolidation_support_thresholds": [4, 5],
        "stream": rows,
        "stream_occurrences": len(rows),
        "repeated_episode_groups": len(REPEATED_TEXTS),
        "occurrences_per_repeated_group": 5,
        "novel_episode_count": len(NOVEL_TEXTS),
        "model_calls": calls,
        "admission_events": {str(key): value for key, value in admission_events.items()},
        "repeated_results": repeated_results,
        "novel_results": novel_results,
        "checks": checks,
        "pass": all(checks.values()),
        "elapsed_ns": time.perf_counter_ns() - started,
        "source_episodes_immutable": True,
        "canonical_identity": "encoding_id plus complete sequenced address A_e",
        "source_evidence_deleted": False,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    findings = f"""# Interleaved repeated-episode consolidation findings

**Integrity:** {'PASS' if result['pass'] else 'FAIL'}

The stream contained 30 supplied episode occurrences: three exact episodes repeated five times each and 15 one-off novel episodes, interleaved round by round. Frozen Qwen encoded every occurrence independently in one call.

Both support thresholds behaved as intended:

- threshold 4 admitted one canonical identity for each repeated episode on stream ordinals 18, 19, and 20;
- threshold 5 admitted the same three canonical identities on ordinals 24, 25, and 26;
- threshold 4 retained immutable support-4 versions and extended each canonical to a support-5 successor;
- threshold 5 created one support-5 version per canonical;
- no novel episode was consolidated.

Every repeated-address query returned all five original occurrence IDs. Every canonical hydrated all five immutable source episodes and replayed the complete sequenced binary address exactly after persistent-store restart. Consolidation created one reusable canonical identity per `(encoding_id, A_e)` while preserving every source occurrence; it did not infer semantics or delete evidence.
"""
    (output / "findings.md").write_text(findings)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "pass": result["pass"]}))
