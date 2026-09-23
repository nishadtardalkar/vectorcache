# VectorCache Algorithm

Approximate nearest-neighbor search matching turbovec's TurboQuant core, with an optional streaming cosine k-means bucketing layer in front of flat FastScan.

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

## Streaming cosine k-means bucketing (`BucketedTurboQuantIndex`)

Clustering is in **input L2-normalized space** (not TurboQuant rotated space). Shared rotation/codebook/TQ+ across buckets.

**Ingest**

1. L2-normalize the vector
2. Assign to nearest cluster centroid by cosine (dot of unit vectors); AVX2 + OpenMP batch scoring
3. Append unit float + TurboQuant-encode into that bucket; online update centroid and variance `E[‖x−c‖²] = 2(1−E[cos])`
4. If `count ≥ min_split_size` and (`variance ≥ var_threshold` **or** `count > max_bucket_size`): spherical k-means (k=2) on **that bucket’s floats only**, re-encode both children, replace parent (no full-index rebuild). Size-based splits keep buckets fine enough that `scan_fraction` is not dominated by one huge open.

**Query**

1. Prepare TurboQuant query state once (rotate + LUT/PD)
2. Score query against all cluster centroids; sort buckets descending
3. Open buckets in order until cumulative size ≥ `scan_fraction * N` (at least one non-empty)
4. FastScan each opened bucket; remap local ids via stored global ids; merge heaps to top-k

Defaults: `scan_fraction=0.1`, `var_threshold=0.5`, `min_split_size=256`, `max_bucket_size=1024`, `split_iters=5`, start with one cluster.

## Ranking

`score = α · Σ_i q_i · centroid[code_i]` (asymmetric IP in rotated / calibrated space).

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
