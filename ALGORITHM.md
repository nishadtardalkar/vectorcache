# VectorCache Algorithm

Two-stage ANN over a flat corpus (`LandmarkIndex`):

**FWHT → quant → orthonormal landmark shortlist → quantized re-rank.**

Defaults: `scan_bits=4`, `rerank_bits=4` (same unless overridden), `m=96` landmarks, shortlist `S=1024`, final top-`k` after re-rank. `dim` must be a positive multiple of 8, ≤ 16384.

## Ingest

For each database vector:

1. **L2-normalize** → unit `x̂` (scratch until `prepare()`)
2. **1 round FWHT** (SRHT stand-in): ChaCha8 Fisher–Yates permutation + ±1 signs + normalized block Walsh–Hadamard. Block size `B = dim & -dim`. Frozen ChaCha8 seed from turbovec v5.
3. **Lloyd-Max scalar quantize** (`rerank_bits`) each rotated coord to one of `2^n` centroids (`n ∈ {2,3,4}`) on Beta((dim−1)/2, (dim−1)/2); boundaries are f32 midpoints of centroids → reconstruction `ŷ`
4. **RaBitQ scale** `α = ⟨y, ŷ⟩ / ‖ŷ‖²` (degenerate → 0)
5. **Landmark project** `ℓ = (α · ŷ) L` where `L` is `dim × m` with orthonormal columns (QR / modified Gram–Schmidt of seeded Gaussians)

At `prepare()`:

- Encode full-dim unit vectors with TurboQuant (`encode`) → store **flat `rerank_bits` codes** + per-vector α; **drop float originals**
- L2-normalize landmark rows, quantize at `scan_bits`, FastScan-pack into **BLOCK=32 SoA** (PERM0)

```
 INGEST                              QUERY
 ──────                              ─────
 float[dim]                          float[dim]
      │                                   │
      ▼                                   ▼
 L2 normalize                    L2 normalize
      │                                   │
      ▼                                   ▼
 FWHT ×1                             FWHT ×1
      │                                   │
      ▼                                   ▼
 4-bit Lloyd-Max + α            project q_rot → q_ℓ (float)
      │                                   │
      ▼                                   ▼
 ℓ = (α·ŷ) L                     AVX2 blocked landmark scan
      │                                   │  heap top-S
      ▼                                   ▼
 prepare():                      quantized re-rank vs codes
  store full-dim codes+α              (α · LUT(q_rot))
  pack landmark BLOCK=32 SoA  ──────→ top-k
  drop floats
```

## Query

1. L2-normalize query → `q̂`
2. Same **1 round FWHT** as ingest → `q_rot`
3. Project onto landmarks: `q_ℓ = q_rot L` (query stays **float**; DB landmarks already baked from quantized `α·ŷ`)
4. **Shortlist scan (FastScan)**: L2-normalize landmark rows, `scan_bits` Lloyd-Max + `α`, bit-plane pack (PERM0 or vector-major); query builds LUT on `q_ℓ` and scores with AVX2 `vpshufb` / AVX-512 VNNI (batched up to 8 queries). Keep top-`S`.
5. **Quantized re-rank**: for each shortlist id, score `α_i · LUT(q_rot, code_i)` against stored full-dim `rerank_bits` codes; heap / sort to final top-`k`

Asymmetric throughout: query stays float (LUT); DB uses `scan_bits` codes for landmark shortlist and `rerank_bits` codes for full-dim re-rank.

## Ranking

| Stage | Score |
|---|---|
| Landmark shortlist | `⟨q_ℓ, ℓ_i⟩` with `ℓ_i = (α_i · ŷ_i) L` (via FastScan LUT) |
| Quantized re-rank | `α_i · ⟨q_rot, ŷ_i⟩` via LUT on full-dim codes |

## Layout / SIMD

- Landmark dim padded to a multiple of 8 for FastScan (`scan_dim`)
- Codes: bit-plane pack → BLOCK=32 **PERM0** (paired with `avx2_perm0` LUT scan)
- Full-dim re-rank codes: row-major flat (`n × n_byte_groups`) for sparse shortlist access
- **Multi-query FastScan** (turbovec-style): up to 4 queries share one codes stream (`kAvx2NqBatch`); OpenMP tiles `(query-batch × block-range)`
- Prior TurboQuant VNNI / vector-major kernels remain in-tree for full-dim paths; landmark search uses PERM0 for pack/search layout match

## CLI (`query-bench`)

```bash
./query-bench --dataset sift1m --bits 4 --landmarks 96 --shortlist 1024 --k 64 --recall
./query-bench --dataset sift1m --scan-bits 2 --rerank-bits 4 --landmarks 96 --shortlist 1024 --k 64 --recall
./query-bench --npy data/corpus.npy --limit 100000 --query-limit 1000 --landmarks 64 --shortlist 8192 --k 64
```

| Flag | Role |
|---|---|
| `--bits` / `--scan-bits` | Landmark FastScan quant bits (2–4) |
| `--rerank-bits` | Full-dim quantized re-rank bits (2–4; default = scan) |
| `--landmarks` / `m` | Orthonormal landmark dimensionality |
| `--shortlist` / `S` | Landmark-space candidate count before re-rank |
| `--k` | Final top-k after quantized re-rank |

## Sources

| Area | Files |
|---|---|
| Index | `include/vectorcache/index.hpp`, `src/index.cpp` |
| FWHT rotation | `include/vectorcache/transform/rotation.hpp`, `src/transform/rotation.cpp` |
| Landmarks (QR) | `include/vectorcache/transform/landmarks.hpp`, `src/transform/landmarks.cpp` |
| Codebook | `include/vectorcache/quantize/codebook.hpp`, `src/quantize/codebook.cpp` |
| Landmark scan + re-rank | `include/vectorcache/search/landmark_scan.hpp`, `src/search/landmark_scan.cpp`, `src/search/search.cpp` |
| FastScan pack / AVX2 / VNNI | `include/vectorcache/pack/pack.hpp`, `src/pack/pack.cpp`, `src/search/search_avx2.cpp`, `src/search/search_vnni.cpp` |
| Encode (landmark FastScan quant) | `include/vectorcache/encode/encode.hpp`, `src/encode/encode.cpp` |
| Top-k heap | `include/vectorcache/search/topk_heap.hpp` |
| Bench | `tools/query-bench.cpp` |
