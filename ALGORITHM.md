# VectorCache Algorithm

Approximate nearest-neighbor search via **TurboVec-style orthogonal rotation**, **TurboQuant** scalar codes (`n` bits per rotated dimension), and **spherical cluster IVF** prune before quantized rescoring.

1. L2-normalize on `dim`
2. Apply `K` rounds of (global Fisher–Yates permutation → ±1 signs → normalized block Walsh–Hadamard); `K` is compile-time (`VECTORCACHE_SRHT_ROUNDS`, default **2**). Block size is the largest power-of-two divisor of `dim` (no zero-pad).
3. **Cluster IVF buckets (prune)** on the rotated vector: maintain `B` unit centroids (random init from `bucket_seed`). Assign each vector to `argmax_j ⟨x, ĉ_j⟩`, then update the exact running mean: `S_j += x`, `n_j++`, `ĉ_j = S_j/‖S_j‖`. Every `rebalance_every` vectors (and once at finalize): one simultaneous frame-potential step minimizing \(\sum_{i<j}\langle ĉ_i,ĉ_j\rangle^2\) with step size `ortho_eta` (default `0.1`; `0` skips), projecting the gradient onto the tangent plane and re-normalizing (non-empty `S_j` are rescaled to the new direction); then reassign all retained post-SRHT floats to the pushed `ĉ` and recompute `S`/`n`/`ĉ` (empty buckets keep prior `ĉ`). Argsort the store by cell key and build a CSR so each cell is a contiguous row range; store final normalized centroids on the index.
4. Lloyd-Max scalar quantize: each rotated coord → one of `2^n` centroids on **Beta((dim−1)/2, (dim−1)/2)** on `[-1,1]`; pack `n`-bit indices (`M = dim` codes)
5. Store per-vector IP scale `α = 1 / ⟨u, x̂⟩` (unit `u`, reconstruction `x̂`) for RaBitQ-style length renormalization
6. Query: same prep (SRHT; keep query float in rotated space for scoring), **nprobe** nearest cells by `⟨q, ĉ_j⟩` (ordered by score desc), score with asymmetric IP × `α`, take top-k

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
        ├─ max-IP cluster assign            ├─ top nprobe by ⟨q, ĉ⟩
        │  + running mean / rebalance       ▼
        ▼                            score × α → top-k
 Lloyd-Max n bits / dim
 + store α = 1/⟨u,x̂⟩
        │
 argsort by cell key + CSR
```

## Ranking

Asymmetric inner product in rotated space with length renormalization:

`score = α · Σ_i q_i · centroid[code_i]`

Higher is better. Only vectors in probed cluster cells are scored.

Codes are stored vector-major (reordered by bucket for contiguous ranges). For byte-aligned widths (`n ∈ {1,2,4,8}` and `(dim·n) % 8 == 0`), search builds exact float query LUTs (one 256-entry table per packed **code** byte-group) and scores each probed CSR range with vector-major LUT / mask-add kernels. Odd / non-byte-aligned widths unpack scalar codes per vector.

| bits | Hot path |
|------|----------|
| 1 | AVX-512 mask-add (`score = base + Σ_{bit b set} Δ_b` over `dim` codes), 4-way interleave of DB rows; requires `dim % 64 == 0` |
| 2 / 4 / 8 | Byte-group float LUT lookup (gather / scalar table) over vector-major packed codes |
| odd | Scalar unpack + MAC |

With `VECTORCACHE_OPENMP`, search parallelizes across probed cells when there are enough cells/candidates **and** `omp_get_max_threads() > 1`: per-thread top-k then merge. The math is unchanged; `α` is applied after the asymmetric IP score.

## Query knobs (`QueryParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `k` | 10 | top-k |
| `probe_radius` | 8 | nprobe: number of nearest cluster lists by centroid IP |

Index-time cluster knobs (`BucketParams` / CLI): `num_buckets` (`B`), `rebalance_every` (`0` ⇒ finalize-only Lloyd pass), `ortho_eta` (frame-potential step before Lloyd; `0` skips), `bucket_seed`. nprobe must be in `1..4096` and is capped by `B` at probe time.

Runtime bits-per-dim (`bits_per_dim` field name) is set at ingest (`IngestionEngine` / `--bits` / `BITS`) and stored on `VectorStore` (`bits` 1–8).

## Primary sources

| Area | Files |
|------|-------|
| Cluster IVF buckets | `include/vectorcache/index/rp_buckets.hpp`, `src/index/rp_buckets.cpp` |
| Quantize | `include/vectorcache/quantize/quantize.hpp`, `src/quantize/quantize.cpp` |
| Rotation | `include/vectorcache/transform/srht.hpp`, `src/transform/srht.cpp` |
| Store | `include/vectorcache/ingest/store.hpp`, `src/ingest/store.cpp` |
| Query | `src/query/engine.cpp`, `src/query/distance.cpp` |
