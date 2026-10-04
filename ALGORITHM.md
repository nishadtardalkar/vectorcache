# VectorCache Algorithm

Approximate nearest-neighbor search matching turbovec's TurboQuant core, with an optional online moving-mean / cosine-variance bucketing layer in front of flat FastScan.

## Per-bucket TurboQuant (unchanged core)

1. L2-normalize on `dim` (8-way chain norm, no FMA)
2. Apply `K=2` rounds of (global ChaCha8 Fisher–Yates → ±1 signs → normalized block Walsh–Hadamard). Block size `B = dim & -dim`. Frozen ChaCha8 seed from turbovec v5.
3. Optional TQ+: per-coord `(x + shift) * scale` fitted from sample quantiles onto outermost Lloyd-Max centroids
4. Lloyd-Max scalar quantize: each rotated coord → one of `2^n` centroids (`n ∈ {2,3,4}`) on Beta((dim−1)/2,(dim−1)/2); boundaries are f32 midpoints of f32 centroids; pack bit-planes
5. Store per-vector scale `α = ‖v‖ / ⟨u, x̂⟩` (RaBitQ-style); degenerate inner → 0
6. Query (inside an opened bucket): same prep; score vectors with nibble LUT / FastScan over BLOCK=32 layout; multiply by `α`; take top-k

```
 INGEST (per bucket)                 QUERY (per opened bucket)
 ───────────────────                 ────────────────────────
 float[dim]                          float[dim]
        │                                   │
        ▼                                   ▼
 L2 normalize                        L2 normalize
        │                                   │
        ▼                                   ▼
 perm+signs+block-WH (×2)            perm+signs+block-WH (×2)
        │                                   │
        ▼                                   ▼
 optional TQ+                        TQ+ inverse + bias
        │                                   │
        ▼                                   ▼
 Lloyd-Max n bits / dim              LUT / FastScan × α → top-k
 + store α
```

## Online moving-mean bucketing (`BucketedTurboQuantIndex`)

Clustering is in **input L2-normalized space** (not TurboQuant rotated space). No SRHT on the clustering path. Shared rotation/codebook/TQ+ across buckets for encode/search only.

Each bucket keeps two centroids:

- **Routing centroid** — exact normalized running mean of members (`sum/count`, then L2-normalize). Used for assignment and query opening.
- **Encode centroid** — frozen at bucket birth (founding unit vector, or normalized mean of a split partition). Residuals `r = x̂ − encode_c` are TurboQuant-encoded against this only.

**Ingest**

1. L2-normalize the vector → `x̂`
2. If no buckets exist: spawn one with encode/routing = `x̂`
3. Else assign to `argmax_j ⟨x̂, routing_c_j⟩`
4. Update Welford stats on `cos = ⟨x̂, routing_c⟩` (before mean update), then `sum += x̂`, `routing = normalize(sum)`
5. Append `x̂`; TurboQuant-encode residual vs frozen encode centroid
6. Split the bucket when `(count ≥ min_bucket_size ∧ cos_var ≥ cos_var_threshold)` or `count ≥ max_bucket_size`:
   - Seed A = routing mean; seed B = furthest member
   - Bipartition by nearer seed (median split by cos-to-A if one side empty)
   - Each child: freeze encode = normalize(mean), recompute routing/Welford, re-encode residuals

**Query**

1. Prepare TurboQuant query state **once** on unit `q̂` (rotate + LUT/PD) — same LUTs for every bucket
2. Score `q̂` against all **routing** centroids; sort buckets descending
3. Open buckets in order until cumulative size ≥ `scan_fraction * N` (at least one non-empty)
4. FastScan each opened bucket with the shared LUTs; push `⟨q̂, encode_c⟩ + α · ⟨q̂, r̂⟩`; remap local ids; merge heaps to top-k

Defaults: `scan_fraction=0.1`, `cos_var_threshold=0.02`, `min_bucket_size=1024`, `max_bucket_size=2048`, start with zero clusters.

## Ranking

Flat: `score = α · Σ_i q_i · centroid[code_i]` (asymmetric IP in rotated / calibrated space).

Bucketed (IVF residual): `score = ⟨q̂, encode_c⟩ + α · Σ_i q_i · centroid[code_i]` where codes quantize residual direction `r̂`.

## Layout

- Bit-plane packed codes → blocked layout (per bucket)
- x86 without AVX-512 VNNI: FAISS `PERM0` hi/lo nibble interleave
- x86 with AVX-512 VNNI+VBMI: vector-major units of 4 byte-groups × 32 vectors
- Search dispatch: AVX-512 VNNI (`vpermb`+`vpdpbusd`) on vector-major when VNNI+VBMI available; else AVX2 PERM0 FastScan (`vpshufb`); else scalar `read_code`

## Sources

| Area | Files |
|------|-------|
| Rotation | `include/vectorcache/transform/rotation.hpp`, `src/transform/rotation.cpp` |
| Codebook | `include/vectorcache/quantize/codebook.hpp`, `src/quantize/codebook.cpp` |
| Encode | `include/vectorcache/encode/encode.hpp`, `src/encode/encode.cpp` |
| Pack | `include/vectorcache/pack/pack.hpp`, `src/pack/pack.cpp` |
| Search | `include/vectorcache/search/search.hpp`, `src/search/search*.cpp` |
| Flat index | `include/vectorcache/index.hpp`, `src/index.cpp` |
| Centroid score | `include/vectorcache/cluster/score_centroids.hpp`, `src/cluster/score_centroids.cpp` |
| Bucketed index | `include/vectorcache/cluster/kmeans_buckets.hpp`, `src/cluster/kmeans_buckets.cpp` |
