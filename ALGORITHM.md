# VectorCache Algorithm

Two-stage ANN: FWHT → 4-bit → orthonormal landmarks shortlist → exact float re-rank.

## Index / query

1. L2-normalize
2. **1 round** FWHT (ChaCha8 Fisher–Yates + ±1 signs + normalized block Walsh–Hadamard, `B = dim & -dim`)
3. Lloyd-Max scalar quantize (`bits` ∈ {2,3,4}) on Beta((dim−1)/2,(dim−1)/2); RaBitQ `α = ⟨y, ŷ⟩ / ‖ŷ‖²`
4. Project `α·ŷ` onto `m` orthonormal landmarks (QR of seeded Gaussians)
5. Pack landmark rows into BLOCK=32 SoA layout; AVX2 FMA shortlist scan + heap
6. Exact IP re-rank of the shortlist against stored unit float vectors → top-k

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

## Online dense bucketing (`BucketedTurboQuantIndex`)

Clustering is in **input L2-normalized space** (not TurboQuant rotated space). No SRHT on the clustering path. Shared rotation/codebook/TQ+ across buckets for encode/search only.

Product clustering matches the SIFT1M online fat champ (`soft309_trig1016`: r@1≈0.981 @ ~3105×322 under `n_probe≈64`).

Each bucket keeps two centroids:

- **Routing centroid** — margin-weighted normalized mean + optional soft seam attract toward near-tie vectors. Used for assignment and query opening. At `prepare()`, thinner lists get a stronger anti-rival push (`C -= β Crival`, β scaled by size asymmetry).
- **Encode centroid** — frozen at bucket birth (founding unit vector, or normalized mean of a split partition). Residuals `r = x̂ − encode_c` are TurboQuant-encoded against this only.

**Ingest**

1. L2-normalize the vector → `x̂`
2. If no buckets exist: spawn one with encode/routing = `x̂`
3. Else assign to `argmax_j ⟨x̂, routing_c_j⟩`; if top-2 margin is thin, soft-pull both centroids
4. Append with margin weight `w = clip(margin/(margin+τ))`; update weighted routing mean; TurboQuant-encode residual vs frozen encode centroid
5. Fission via 1-iter spherical 2-means poles + Voronoi membership (median-axis fallback if a side has <2):
   - **Hard** when `count ≥ max_bucket_size` (default 800)
   - **Soft** when `count ≥ soft_t` and `energy = 1−cos_mean ≥ trig` (`energy_soft=400` / `energy_trig=0.095`, with mid-stream densify pulse when `expected_n` is set)
6. Periodic absorb of tiny near-duplicate lists into larger rivals (stream absorb / recovery / late)

**Query**

1. Prepare TurboQuant query state **once** on unit `q̂` (rotate + LUT/PD) — same LUTs for every bucket
2. Score `q̂` against all **routing** centroids; sort buckets descending
3. Open the top `n_probe` buckets by routing score (capped at `num_buckets`)
4. FastScan each opened bucket with the shared LUTs; push `⟨q̂, encode_c⟩ + α · ⟨q̂, r̂⟩`; remap local ids; merge heaps to top-k

Defaults: `n_probe=64`, `max_bucket_size=800`, `energy_soft=400`, pulse densify when `expected_n>0` (fractions 0.65/0.78/0.90 of N).

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
