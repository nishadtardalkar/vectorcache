# VectorCache Algorithm

Approximate nearest-neighbor search via **TurboVec-style orthogonal rotation** and **TurboQuant** block codes (`n` bits per block of `d` dimensions; default `d=1`) with a flat full-corpus scan.

1. L2-normalize on `dim`
2. Apply `K` rounds of (global Fisher–Yates permutation → ±1 signs → normalized block Walsh–Hadamard); `K` is compile-time (`VECTORCACHE_SRHT_ROUNDS`, default **2**). Block size is the largest power-of-two divisor of `dim` (no zero-pad).
3. Lloyd-Max / block-VQ: each contiguous block of `d` rotated coords → one of `2^n` centroids in `R^d` for **Beta((dim−1)/2, (dim−1)/2)** (product measure when `d>1`); pack `n`-bit indices (`M = dim/d` codes)
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
 Lloyd-Max n bits / d dims      (rotated float query)
 + store α = 1/⟨u,x̂⟩                    │
        │                              ▼
 append to flat store           score ALL codes × α → top-k
```

## Ranking

Asymmetric inner product in rotated space with length renormalization:

`score = α · Σ_b ⟨q_block_b, centroid[code_b]⟩`

Higher is better. Same path for all `(d, n)`.

For byte-aligned widths (`n ∈ {1,2,4,8}` and `(M·n) % 8 == 0`), search builds exact float query LUTs (one 256-entry table per packed **code** byte-group) and scores from a **FAISS FastScan-style blocked cache** (`BLOCK=32`: for each byte-group, 32 vectors’ bytes are contiguous). Canonical ingest storage remains vector-major packed codes; the blocked view is rebuilt lazily on search (or via `QueryEngine::prepare_index()`).

| bits | Hot path |
|------|----------|
| 1 | Transposed u64 columns + AVX-512 mask-add (`score = base + Σ_{bit b set} Δ_b` over `M` codes), 4-way interleave within each block; requires `M % 64 == 0` |
| 4 | Nibble-split float LUTs + `_mm512_permutexvar_ps` (exact) |
| 8 | AVX-512 gather from 256-entry float LUTs across 32 lanes (exact); LUT fill specialized for one code/group (`d=2` uses FMA over centroids) |
| 2 | Blocked float LUT lookup across 32 lanes |
| odd | Scalar unpack + block MAC (no blocked cache) |

With `VECTORCACHE_OPENMP`, the block loop parallelizes when `n_blocks ≥ 1024` (~32K vectors) **and** `omp_get_max_threads() > 1`: per-thread top-k then merge. Byte-aligned widths always score from the blocked cache when the query LUT is non-empty (single-thread or parallel); odd / non-byte-aligned widths fall back to vector-major scalar unpack. The math is unchanged; `α` is applied after the block score.

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |

Runtime bits-per-block (`bits_per_dim` field name kept for compatibility) and `block_dims` are set at ingest (`IngestionEngine` / `--bits` / `--block-dims` / `BITS` / `BLOCK_DIMS`) and stored on `VectorStore` (`bits` 1–8, `block_dims` 1–16, `srht_dim % block_dims == 0`).

## Primary sources

| Area | Files |
|------|-------|
| Quantize | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| Rotation | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp`, `src/query/fastscan.cpp` |
