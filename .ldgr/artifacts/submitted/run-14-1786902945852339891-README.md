---
pretty_name: Embedding Exact Recurrence — Interleaved Episode Consolidation
tags:
  - episodic-memory
  - recurrence
  - consolidation
  - qwen
  - population-coding
---

# Embedding Exact Recurrence

Complete evidence for the exact repeated-episode consolidation trial in
[`hydra-dynamix/embedding-exact-recurrance`](https://github.com/hydra-dynamix/embedding-exact-recurrance).

## Contents

- 30 independently encoded episode occurrences;
- three exact episodes repeated five times each;
- 15 interleaved one-off novel episodes;
- uncompressed input, token, `H`, `N`, and `B = A_e` evidence;
- persistent collision-safe episode indexes;
- support-threshold 4 and 5 canonical catalogs;
- complete results, findings, and independent audit;
- `full-data-manifest.json` with SHA-256 and byte size for every uploaded file.

## Result

Both occurrence-support thresholds created exactly one canonical identity for each repeated `(encoding_id, A_e)` and promoted none of the novel episodes. Every repeated query returned all five immutable source occurrences, and every canonical hydrated and replayed exactly after restart.

Threshold 4 admitted on the fourth occurrence and retained immutable support-4 and support-5 versions. Threshold 5 admitted on the fifth occurrence and retained one support-5 version.

## Frozen representation

```text
model: Qwen/Qwen3-Embedding-0.6B
revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
normalization: row-minmax-v1
representation threshold: 0.8
consolidation occurrence-support thresholds: 4 and 5
```

The texts are synthetic. Source episodes remain directly addressable; canonical records contain bindings rather than replacing or deleting evidence.
