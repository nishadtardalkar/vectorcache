# VectorCache Algorithm

Approximate nearest-neighbor search with **top-d support keys** for candidate generation and **L0 1-bit codes** for ranking.

1. L2-normalize on true `input_dim` (no pad in the key)
2. Support key = sorted indices of the `d` largest `|x_i|` (default `d=4`)
3. Zero-pad **only inside SRHT** to `srht_dim = next_pow2(input_dim)` for FWHT
4. Store L0 sign bits on the SRHT output
5. Query: score every unique support key by dim intersection with the query key, scan top `n_buckets`, then L0 bit-agreement top-k

```
 INGEST                                              QUERY
 ──────                                              ─────
 float[input_dim]                                    float[input_dim]
        │                                                   │
        ▼                                                   ▼
 L2 normalize                                        L2 normalize
        │                                                   │
        ├──────────────────────┐                            ├────────────────┐
        ▼                      ▼                            ▼                ▼
 top-d support key      pad + SRHT                 support key         pad + SRHT
        │                      │                            │                │
        │                      ▼                            │                ▼
        │                 L0 sign bits                      │           L0 sign bits
        └──────────┬───────────┘                            └───────┬────────┘
                   ▼                                                ▼
         hash(key) → bucket                              score all keys vs query
         (ids + L0)                                      take top n_buckets, L0 top-k
```

## Support key

- Depth `d` ∈ `[1, 16]` (`kDefaultSupportDepth = 4`)
- Tie-break when `|x|` equal: **higher index wins**
- Hamming distance between equal-weight keys: `HD = 2*(d − |intersection|)`
- One posting per vector (no multipost). Recall via ranking all keys and scanning top buckets.

## Padding

Padding exists **only** so FWHT can run at power-of-two length. Support keys never see pad zeros. L0 width equals `srht_dim`.

## Ranking

`disagree = popcount(q ⊕ d)`, score `(B − 2·disagree)/B` with `B = srht_dim`. Same AVX-512 path as before.

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |
| `n_buckets` | 32 | after scoring every unique support key by intersection with the query key, scan this many best buckets |

## Primary sources

| Area | Files |
|------|-------|
| Support + L0 | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| HD postings | `include/vectorcache/quantize/dim_postings.hpp`, `src/quantize/dim_postings.cpp` |
| SRHT | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp` |
