# VectorCache Algorithm

Two-stage ANN over a flat corpus (`LandmarkIndex`):

**FWHT → 4-bit quant → orthonormal landmark shortlist → exact float re-rank.**

Defaults: `bits=4`, `m=96` landmarks, shortlist `S=1024`, final top-`k` after re-rank. `dim` must be a positive multiple of 8, ≤ 16384.

## Ingest

For each database vector:

1. **L2-normalize** → unit `x̂` (stored for exact re-rank)
2. **1 round FWHT** (SRHT stand-in): ChaCha8 Fisher–Yates permutation + ±1 signs + normalized block Walsh–Hadamard. Block size `B = dim & -dim`. Frozen ChaCha8 seed from turbovec v5.
3. **Lloyd-Max scalar quantize** each rotated coord to one of `2^n` centroids (`n ∈ {2,3,4}`) on Beta((dim−1)/2, (dim−1)/2); boundaries are f32 midpoints of centroids → reconstruction `ŷ`
4. **RaBitQ scale** `α = ⟨y, ŷ⟩ / ‖ŷ‖²` (degenerate → 0)
5. **Landmark project** `ℓ = (α · ŷ) L` where `L` is `dim × m` with orthonormal columns (QR / modified Gram–Schmidt of seeded Gaussians)

At `prepare()`: pack landmark rows into **BLOCK=32 SoA** layout (per block: for each of `m` dims, 32 lane floats).

```
 INGEST                              QUERY
 ──────                              ─────
 float[dim]                          float[dim]
      │                                   │
      ▼                                   ▼
 L2 normalize  ──store x̂──┐         L2 normalize
      │                   │               │
      ▼                   │               ▼
 FWHT ×1                  │          FWHT ×1
      │                   │               │
      ▼                   │               ▼
 4-bit Lloyd-Max + α      │          project q_rot → q_ℓ (float)
      │                   │               │
      ▼                   │               ▼
 ℓ = (α·ŷ) L              │          AVX2 blocked landmark scan
      │                   │               │  heap top-S
      ▼                   │               ▼
 pack BLOCK=32 SoA        │          exact IP re-rank vs x̂
                          └──────────→ top-k
```

## Query

1. L2-normalize query → `q̂`
2. Same **1 round FWHT** as ingest → `q_rot`
3. Project onto landmarks: `q_ℓ = q_rot L` (query stays **float**; DB landmarks already baked from quantized `α·ŷ`)
4. **Shortlist scan**: score `⟨q_ℓ, ℓ_i⟩` over all DB landmark rows (AVX2 FMA over BLOCK=32 SoA); keep top-`S` with a min-heap
5. **Exact re-rank**: for each shortlist id, score `⟨q̂, x̂_i⟩` in original unit space; heap / sort to final top-`k`

Asymmetric: only the DB path is 4-bit quantized; the query is never quantized. Landmark scan is float·float against the dequantized-then-projected DB features (not Hamming).

## Ranking

| Stage | Score |
|---|---|
| Landmark shortlist | `⟨q_ℓ, ℓ_i⟩` with `ℓ_i = (α_i · ŷ_i) L` |
| Exact re-rank | `⟨q̂, x̂_i⟩` (unit IP ≡ cosine) |

## Layout / SIMD

- Landmark DB: blocked SoA, `n_blocks × m × 32` floats
- Shortlist kernel: broadcast each `q_ℓ[d]`, `_mm256_fmadd_ps` across 32 lanes (4× YMM), heap push
- OpenMP over queries
- Backend name: `landmark-avx2+rerank`

## CLI (`query-bench`)

```bash
./query-bench --dataset sift1m --bits 4 --landmarks 96 --shortlist 1024 --k 64 --recall
./query-bench --npy data/corpus.npy --limit 100000 --query-limit 1000 --landmarks 64 --shortlist 8192 --k 64
```

| Flag | Role |
|---|---|
| `--bits` | Scalar quant bits before landmarks (2–4) |
| `--landmarks` / `m` | Orthonormal landmark dimensionality |
| `--shortlist` / `S` | Landmark-space candidate count before re-rank |
| `--k` | Final top-k after exact re-rank |

## Sources

| Area | Files |
|---|---|
| Index | `include/vectorcache/index.hpp`, `src/index.cpp` |
| FWHT rotation | `include/vectorcache/transform/rotation.hpp`, `src/transform/rotation.cpp` |
| Landmarks (QR) | `include/vectorcache/transform/landmarks.hpp`, `src/transform/landmarks.cpp` |
| Codebook | `include/vectorcache/quantize/codebook.hpp`, `src/quantize/codebook.cpp` |
| Landmark scan + re-rank | `include/vectorcache/search/landmark_scan.hpp`, `src/search/landmark_scan.cpp`, `src/search/search.cpp` |
| Top-k heap | `include/vectorcache/search/topk_heap.hpp` |
| Bench | `tools/query-bench.cpp` |
