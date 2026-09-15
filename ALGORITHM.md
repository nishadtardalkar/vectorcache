# VectorCache Algorithm

Approximate nearest-neighbor search via **TurboVec-style orthogonal rotation** and **TurboQuant** scalar codes (`n` bits per dimension) with a flat full-corpus scan.

1. L2-normalize on `dim`
2. Apply `K` rounds of (global Fisher–Yates permutation → ±1 signs → normalized block Walsh–Hadamard); `K` is compile-time (`VECTORCACHE_SRHT_ROUNDS`, default **2**). Block size is the largest power-of-two divisor of `dim` (no zero-pad).
3. Lloyd-Max scalar quantize each rotated coordinate to `2^n` centroids for **Beta((dim−1)/2, (dim−1)/2)** on `[-1, 1]`; pack `n`-bit indices
4. Store per-vector IP scale `α = 1 / ⟨u, x̂⟩` (unit `u`, reconstruction `x̂`) for RaBitQ-style length renormalization
5. Query: same prep (keep query float in rotated space), score **every** stored code with asymmetric IP, multiply by `α`, take top-k

```
 INGEST                         QUERY
 ──────                         ─────
 float[dim]                     float[dim]
        │                              │
        ▼                              ▼
 L2 normalize                   L2 normalize
        │                              │
        ▼                              ▼
 perm+signs+block-WH (×K)       perm+signs+block-WH (×K)
        │                              │
        ▼                              ▼
 Lloyd-Max n bits/dim           (rotated float query)
 + store α = 1/⟨u,x̂⟩                    │
        │                              ▼
 append to flat store           score ALL codes × α → top-k
```

## Ranking

Asymmetric inner product in rotated space with length renormalization:

`score = α · Σ_i q_rot[i] · centroid[code_i]`

Higher is better. Same path for all `n`.

For byte-aligned widths (`n ∈ {1,2,4,8}`), search builds exact float query LUTs (one 256-entry table per packed byte-group) and scores from a **FAISS FastScan-style blocked cache** (`BLOCK=32`: for each byte-group, 32 vectors’ bytes are contiguous). Canonical ingest storage remains vector-major packed codes; the blocked view is rebuilt lazily on search (or via `QueryEngine::prepare_index()`).

| bits | Hot path |
|------|----------|
| 1 | Transposed u64 columns + AVX-512 mask-add (`score = base + Σ_{bit i set} Δ_i`), 4-way interleave within each block |
| 4 | Nibble-split float LUTs + `_mm512_permutexvar_ps` (exact) |
| 2 / 8 | Blocked float LUT lookup across 32 lanes |
| odd | Scalar unpack + MAC (no blocked cache) |

With `VECTORCACHE_OPENMP`, the block loop parallelizes when `n_blocks ≥ 1024` (~32K vectors) **and** `omp_get_max_threads() > 1`: per-thread top-k then merge. Single-thread / smaller corpora use the existing vector-major LUT/mask-add kernels (blocked layout is still built for bits=1 mask-add when `dim % 64 == 0`). The math is unchanged; `α` is applied after the block score.

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |

Runtime bits-per-dim is set at ingest (`IngestionEngine` / `--bits` / `BITS`) and stored on `VectorStore` (range 1–8).

## Primary sources

| Area | Files |
|------|-------|
| Quantize | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| Rotation | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp`, `src/query/fastscan.cpp` |
