# VectorCache Algorithm

Flat approximate nearest-neighbor search matching turbovec's TurboQuant core (no IVF).

1. L2-normalize on `dim` (8-way chain norm, no FMA)
2. Apply `K=2` rounds of (global ChaCha8 Fisher–Yates → ±1 signs → normalized block Walsh–Hadamard). Block size `B = dim & -dim`. Frozen ChaCha8 seed from turbovec v5.
3. Optional TQ+: per-coord `(x + shift) * scale` fitted from sample quantiles onto outermost Lloyd-Max centroids
4. Lloyd-Max scalar quantize: each rotated coord → one of `2^n` centroids (`n ∈ {2,3,4}`) on Beta((dim−1)/2,(dim−1)/2); boundaries are f32 midpoints of f32 centroids; pack bit-planes
5. Store per-vector scale `α = ‖v‖ / ⟨u, x̂⟩` (RaBitQ-style); degenerate inner → 0
6. Query: same prep; score all vectors with nibble LUT / FastScan over BLOCK=32 layout; multiply by `α`; take top-k

```
 INGEST                              QUERY
 ──────                              ─────
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

## Ranking

`score = α · Σ_i q_i · centroid[code_i]` (asymmetric IP in rotated / calibrated space).

## Layout

- Bit-plane packed codes → blocked layout
- x86 without AVX-512 VNNI: FAISS `PERM0` hi/lo nibble interleave
- x86 with AVX-512 VNNI+VBMI: vector-major units of 4 byte-groups × 32 vectors
- Search dispatch: AVX-512 VNNI (`vpermb`+`vpdpbusd`) on vector-major when available; else AVX2 PERM0 FastScan; else scalar `read_code`

## Sources

| Area | Files |
|------|-------|
| Rotation | `include/vectorcache/transform/rotation.hpp`, `src/transform/rotation.cpp` |
| Codebook | `include/vectorcache/quantize/codebook.hpp`, `src/quantize/codebook.cpp` |
| Encode | `include/vectorcache/encode/encode.hpp`, `src/encode/encode.cpp` |
| Pack | `include/vectorcache/pack/pack.hpp`, `src/pack/pack.cpp` |
| Search | `include/vectorcache/search/search.hpp`, `src/search/search*.cpp` |
| Index | `include/vectorcache/index.hpp`, `src/index.cpp` |
