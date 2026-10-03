# VectorCache Algorithm

Approximate nearest-neighbor search matching turbovec's TurboQuant core, with an optional online cosine-threshold bucketing layer in front of flat FastScan.

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

## Online cosine-threshold bucketing (`BucketedTurboQuantIndex`)

Clustering is in **input L2-normalized space** (not TurboQuant rotated space). No SRHT on the clustering path. Shared rotation/codebook/TQ+ across buckets for encode/search only.

**Ingest**

1. L2-normalize the vector → `x̂`
2. Among existing buckets with `⟨x̂, c⟩ ≥ cos_threshold`, pick the spherical-FSCL winner  
   `argmax_j (target_bucket_size / n_j) · ⟨x̂, c_j⟩ − log(n_j)` (natural log; Banerjee & Ghosh with `N/k` → `target_bucket_size`)
3. If no joinable bucket: create a new cluster with the current unit vector as the **frozen** centroid, then append
4. Else append into the FSCL winner (centroid unchanged)
5. Form residual `r = x̂ − c`; TurboQuant-encode `r` into the chosen bucket (`α` recovers `‖r‖` via the usual RaBitQ scale)

**Query**

1. Prepare TurboQuant query state **once** on unit `q̂` (rotate + LUT/PD) — same LUTs for every bucket
2. Score `q̂` against all cluster centroids; sort buckets descending
3. Open buckets in order until cumulative size ≥ `scan_fraction * N` (at least one non-empty)
4. FastScan each opened bucket with the shared LUTs; push `⟨q̂, c⟩ + α · ⟨q̂, r̂⟩`; remap local ids; merge heaps to top-k

Defaults: `scan_fraction=0.1`, `cos_threshold=0.7`, `target_bucket_size=20000`, start with zero clusters.

## Ranking

Flat: `score = α · Σ_i q_i · centroid[code_i]` (asymmetric IP in rotated / calibrated space).

Bucketed (IVF residual): `score = ⟨q̂, c⟩ + α · Σ_i q_i · centroid[code_i]` where codes quantize residual direction `r̂`.

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
