# VectorCache Algorithm

Approximate nearest-neighbor search via **SRHT** rotation and **L0 1-bit** codes with a flat full-corpus scan.

1. L2-normalize on true `input_dim`
2. Zero-pad to `srht_dim = next_pow2(input_dim)` for FWHT
3. Apply SRHT (`H·D` rounds, compile-time count)
4. Store L0 sign bits on the SRHT output
5. Query: same prep, then score **every** stored L0 code with bit-agreement top-k

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
 L0 sign bits                   L0 sign bits
        │                              │
        ▼                              ▼
 append to flat store           score ALL codes → top-k
```

## Padding

Padding exists **only** so FWHT can run at power-of-two length. L0 width equals `srht_dim`.

## Ranking

`disagree = popcount(q ⊕ d)`, score `(B − 2·disagree)/B` with `B = srht_dim`. AVX-512 XOR-popcount path.

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |

## Primary sources

| Area | Files |
|------|-------|
| L0 | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| SRHT | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp` |
