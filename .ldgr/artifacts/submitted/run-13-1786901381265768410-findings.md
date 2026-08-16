# Interleaved repeated-episode consolidation findings

**Integrity:** PASS

The stream contained 30 supplied episode occurrences: three exact episodes repeated five times each and 15 one-off novel episodes, interleaved round by round. Frozen Qwen encoded every occurrence independently in one call.

Both support thresholds behaved as intended:

- threshold 4 admitted one canonical identity for each repeated episode on stream ordinals 18, 19, and 20;
- threshold 5 admitted the same three canonical identities on ordinals 24, 25, and 26;
- threshold 4 retained immutable support-4 versions and extended each canonical to a support-5 successor;
- threshold 5 created one support-5 version per canonical;
- no novel episode was consolidated.

Every repeated-address query returned all five original occurrence IDs. Every canonical hydrated all five immutable source episodes and replayed the complete sequenced binary address exactly after persistent-store restart. Consolidation created one reusable canonical identity per `(encoding_id, A_e)` while preserving every source occurrence; it did not infer semantics or delete evidence.
