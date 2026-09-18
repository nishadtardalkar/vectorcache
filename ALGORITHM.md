# VectorCache Algorithm

Approximate nearest-neighbor search via **TurboVec-style orthogonal rotation**, **TurboQuant** block codes (`n` bits per block of `d` dimensions; default `d=1`), and **random-projection bucket prune** before quantized rescoring.

1. L2-normalize on `dim`
2. Apply `K` rounds of (global Fisher–Yates permutation → ±1 signs → normalized block Walsh–Hadamard); `K` is compile-time (`VECTORCACHE_SRHT_ROUNDS`, default **2**). Block size is the largest power-of-two divisor of `dim` (no zero-pad).
3. **RP buckets (prune):** project the rotated unit vector onto `R` independent random unit directions; bin with width `w` as `b_i = ⌊⟨x,r_i⟩ / w⌋`; pack the R-tuple into a cell key. After ingest, argsort the store by cell key and build a CSR inverted index so each cell is a contiguous row range.
4. Lloyd-Max / block-VQ: each contiguous block of `d` rotated coords → one of `2^n` centroids in `R^d` for **Beta((dim−1)/2, (dim−1)/2)** (product measure when `d>1`); pack `n`-bit indices (`M = dim/d` codes)
5. Store per-vector IP scale `α = 1 / ⟨u, x̂⟩` (unit `u`, reconstruction `x̂`) for RaBitQ-style length renormalization
6. Query: same prep (keep query float in rotated space), compute query bins, **multi-probe** all cells with L_∞ offset `≤ P` (ordered by `Σ o_i²`), score only those contiguous ranges with asymmetric IP × `α`, take top-k

```
 INGEST                              QUERY
 ──────                              ─────
 float[dim]                          float[dim]
        │                                   │
        ▼                                   ▼
 L2 normalize                        L2 normalize
        │                                   │
        ▼                                   ▼
 perm+signs+block-WH (×K)            perm+signs+block-WH (×K)
        │                                   │
        ├─ RP dots → bins → cell key        ├─ RP dots → bins
        │                                   ▼
        ▼                            multi-probe cells (L_∞ ≤ P)
 Lloyd-Max n bits / d dims                  │
 + store α = 1/⟨u,x̂⟩                         ▼
        │                            score probed ranges × α → top-k
 argsort by cell key + CSR
```

## Ranking

Asymmetric inner product in rotated space with length renormalization:

`score = α · Σ_b ⟨q_block_b, centroid[code_b]⟩`

Higher is better. Same path for all `(d, n)`. Only vectors in probed RP cells are scored.

For byte-aligned widths (`n ∈ {1,2,4,8}` and `(M·n) % 8 == 0`), search builds exact float query LUTs (one 256-entry table per packed **code** byte-group) and scores from a **FAISS FastScan-style blocked cache** (`BLOCK=32`: for each byte-group, 32 vectors’ bytes are contiguous). Canonical ingest storage remains vector-major packed codes (reordered by bucket); the blocked view is rebuilt lazily on search (or via `QueryEngine::prepare_index()`). Range scoring walks FastScan blocks that intersect each probed `[lo, hi)` and masks partial edge blocks.

| bits | Hot path |
|------|----------|
| 1 | Transposed u64 columns + AVX-512 mask-add (`score = base + Σ_{bit b set} Δ_b` over `M` codes), 4-way interleave within each block; requires `M % 64 == 0` |
| 4 | Nibble-split float LUTs + `_mm512_permutexvar_ps` (exact) |
| 8 | AVX-512 gather from 256-entry float LUTs across 32 lanes (exact); LUT fill specialized for one code/group (`d=2` uses FMA over centroids) |
| 2 | Blocked float LUT lookup across 32 lanes |
| odd | Scalar unpack + block MAC (no blocked cache) |

With `VECTORCACHE_OPENMP`, search parallelizes across probed cells when there are enough cells/candidates **and** `omp_get_max_threads() > 1`: per-thread top-k then merge. Odd / non-byte-aligned widths fall back to vector-major scalar unpack on each range. The math is unchanged; `α` is applied after the block score.

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |
| `probe_radius` | 1 | multi-probe L_∞ radius `P` in bin units |

Index-time RP knobs (`BucketParams` / CLI): `num_projections` (`R`), `bin_width` (`w`), `bucket_seed`. Probe grid size `(2P+1)^R` must be `≤ 4096`.

Runtime bits-per-block (`bits_per_dim` field name kept for compatibility) and `block_dims` are set at ingest (`IngestionEngine` / `--bits` / `--block-dims` / `BITS` / `BLOCK_DIMS`) and stored on `VectorStore` (`bits` 1–8, `block_dims` 1–16, `srht_dim % block_dims == 0`).

## Primary sources

| Area | Files |
|------|-------|
| RP buckets | `include/vectorcache/index/rp_buckets.hpp`, `src/index/rp_buckets.cpp` |
| Quantize | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| Rotation | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp`, `src/query/fastscan.cpp` |
