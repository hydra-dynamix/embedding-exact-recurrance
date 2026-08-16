# Embedding sequencing

A frozen transformer represents each complete input in one pass. Every attended final-layer contextual state becomes one row of an ordered sequence. A detector bank contains one provenance-linked raw reference vector for every admitted memory contextual state.

Detector responses use literal raw dot products in float32:

```text
S[T,M] = H[T,D] @ W[M,D].T
R[T,M] = S[T,M] - tau[M]
B[T,M] = R[T,M] >= 0
```

No L2 normalization, cosine response, pooling, top-k selection, recurrence rule, or construction-admission rule is applied. Debug ranking only limits display and does not alter the complete population.

Run the tests with:

```bash
.venv/bin/python -m unittest discover -v
```
