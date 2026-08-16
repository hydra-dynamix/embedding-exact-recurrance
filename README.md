# Embedding sequencing

Published as [`hydra-dynamix/embedding-exact-recurrance`](https://github.com/hydra-dynamix/embedding-exact-recurrance). Complete real-Qwen consolidation evidence is hosted in the Hugging Face dataset [`bakobiibizo/embedding-exact-recurrance`](https://huggingface.co/datasets/bakobiibizo/embedding-exact-recurrance).

This project encodes a complete text episode into an ordered sequence of binary contextual states and stores its complete in-process evidence under that sequenced address. It does not compare the episode with stored memories, detector prototypes, or reference vectors during initial encoding.

## State and address terminology

A frozen transformer represents the complete text block in one pass and retains every attended final-layer contextual state:

```text
H_t[D] = raw contextual state at position t
```

Each `H_t` is independently min-max normalized across its embedding dimensions:

```text
N_t[d] = (H_t[d] - min(H_t)) / (max(H_t) - min(H_t))
```

The normalized state is thresholded into one binary state array:

```text
B_t[d] = 1 if N_t[d] >= tau_threshold else 0
```

A constant `H_t` has no relatively strongest dimensions and produces an all-zero `N_t` and `B_t`. Every `N_t` remains in `[0,1]`; every `B_t` is a `uint8` vector containing only zero and one.

The episode's **sequenced binary address** is the ordered collection of its binary states:

```text
A_e = (B_0, B_1, ..., B_(T-1))
```

`binary_state_sequence` is the physical `[T,D]` matrix representing `A_e`. `B_t` names one row, not the whole address. Address identity includes the state count, state dimension count, bit order, and every binary value.

## Episode evidence storage

`EpisodicMemoryStore` snapshots the complete current `EncodedEpisode` evidence under `A_e`:

```text
input IDs
attention mask
contextual positions
H[T,D]
N[T,D]
binary_state_sequence[T,D] = A_e
```

If several occurrences produce the same `A_e`, the store retains all of them in one collision bucket. Each occurrence also has a unique `episode_id` for direct hydration. The evidence is stored once; the sequenced-address and episode-ID maps are access paths to the same stored occurrence.

With `--store-root`, `PersistentEpisodicMemoryStore` writes exact input bytes and uncompressed input IDs, masks, positions, `H`, `N`, and `A_e` tensors beneath the address digest. Each occurrence records pinned model/tokenizer/layer/tau/normalization provenance and a chronological store ordinal. Hash, shape, dtype, address, and provenance checks run during restart hydration.

## Local recurrence

`LocalRecurrenceIndex` enumerates every legal contiguous subsequence inside each stored episode boundary:

```text
(B_i, ..., B_(i+L-1)) -> every (episode_id, start_offset, length)
```

By default it indexes all lengths from one through the complete episode length. Exact keys include encoder identity, length, dimensions, order, and all bits. A recurrent candidate requires support from at least two distinct episodes, but every supporting occurrence—including duplicates within an episode—is retained. No result cap, ranking, labels, or cross-boundary window is used.

## Lossless construction factoring

For an explicitly bound group of same-shape occurrence windows, consolidation computes the position-wise common binary component `C` and an exact residual for each occurrence:

```text
B_window(e) = C union residual(e)
C intersect residual(e) = empty
```

Every source window reconstructs exactly. Candidate versions retain complete occurrence bindings, encoder identity, recurrence-key provenance, residuals, and predecessor version references. Exact recurrent windows form the initial zero-residual arm. Candidates are never automatically admitted as constructions; admission remains a separate authority boundary.

## Exact repeated-episode consolidation

`ExactEpisodeConsolidator` observes complete stored episodes and admits one canonical identity for an exact `(encoding_id, A_e)` only after a declared occurrence-support threshold. Source episodes remain immutable. Threshold 4 creates an immutable support-4 version and later occurrences create successor versions under the same canonical identity; threshold 5 waits for the fifth occurrence. One-off novel addresses remain unconsolidated.

Run the interleaved repeated/novel real-Qwen trial with:

```bash
.venv/bin/python experiments/repeated_episode_consolidation.py \
  --output artifacts/interleaved-repeated-episode-consolidation
.venv/bin/python experiments/audit_repeated_episode_consolidation.py \
  --root artifacts/interleaved-repeated-episode-consolidation
```

This is whole-episode consolidation under supplied episode boundaries. It does not use sequence-order nulls, semantic labels, local-window chance recurrence, or construction similarity.

## Run

Encode, store, and retrieve one or more text blocks:

```bash
.venv/bin/python main.py --tau-threshold 0.8 "dog" "a second text block"

# Persist evidence and hydrate it on later runs:
.venv/bin/python main.py --store-root ./memory --tau-threshold 0.8 "dog" "dog"
```

The output contains each `binary_state_sequence`, its sequenced-address shape, every episode ID in the exact-address collision bucket, the complete local-window count, and every recurrent/construction candidate. Raw `H` and normalized `N` are optional diagnostics:

```bash
.venv/bin/python main.py --tau-threshold 0.8 --debug-encoding "dog"
```

Run the tests with:

```bash
.venv/bin/python -m unittest discover -v
```
