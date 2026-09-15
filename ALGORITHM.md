# VectorCache Algorithm

Approximate nearest-neighbor search via **SRHT** rotation and **TurboQuantMSE** codes (`n` bits per dimension) with a flat full-corpus scan.

1. L2-normalize on true `input_dim`
2. Zero-pad to `srht_dim = next_pow2(input_dim)` for FWHT
3. Apply SRHT (`H·D` rounds, compile-time count)
4. Lloyd-Max scalar quantize each SRHT coordinate to `2^n` centroids for `N(0, 1/srht_dim)`; pack `n`-bit indices
5. Query: same prep (keep query float in rotated space), then score **every** stored code with asymmetric IP top-k

```
 INGEST                         QUERY
 ──────                         ─────
 float[input_dim]               float[input_dim]
        │                              │
        ▼                              ▼
 L2 normalize                   L2 normalize
        │                              │
        ▼                              ▼
 pad + SRHT                     pad + SRHT
        │                              │
        ▼                              ▼
 Lloyd-Max n bits/dim           (rotated float query)
        │                              │
        ▼                              ▼
 append to flat store           score ALL codes → top-k
```

## Padding

Padding exists **only** so FWHT can run at power-of-two length. Code width is `srht_dim * n` bits.

## Ranking

Asymmetric inner product in rotated space:

`score = Σ_i q_rot[i] · centroid[code_i]`

Higher is better. Same path for all `n` (including `n = 1`, where centroids are `±√(2/(π·d))`).

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |

Runtime bits-per-dim is set at ingest (`IngestionEngine` / `--bits` / `BITS`) and stored on `VectorStore` (range 1–8).

## Primary sources

| Area | Files |
|------|-------|
| Quantize | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| SRHT | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp` |
