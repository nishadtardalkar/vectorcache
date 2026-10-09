# VectorCache

TurboQuant ANN engine (C++20) matching [turbovec](https://github.com/RyanCodrai/turbovec)'s core path: ChaCha8 block-Hadamard rotation, Lloyd-Max Beta codebook, bit-plane encode, FastScan blocked layout, and SIMD search — plus an optional online moving-mean / cosine-variance bucketing layer and dataset fetch helpers for TurboVec/TurboQuant benchmarks.

## Requirements

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.20+
- OpenMP (recommended)
- For `fetch-datasets`: libcurl; Apache Arrow/Parquet for OpenAI datasets
- For GloVe / SIFT1M: HDF5 C library

On HPC clusters:

```bash
source scripts/envs.sh
```

## HPC workflow

```bash
# Login node (internet):
make login
make login DATASETS=glove
make login DATASETS=sift1m

# Compute node (offline):
make compute
make compute BITS=4 RECALL=1 CALIBRATE=1
make compute DATASET=openai-1536 BITS=2 K=64
make compute DATASET=sift1m LIMIT=0 QUERY_LIMIT=10000
```

`make login` configures CMake and downloads datasets into `data/`.  
`make compute` reconfigures offline, builds tests + `query-bench`, runs `ctest`, then runs `query-bench`.

## Algorithm (turbovec-compatible + optional IVF)

1. L2-normalize
2. K=2 rounds of global ChaCha8 Fisher–Yates → ±1 signs → normalized block Walsh–Hadamard (`B = dim & -dim`)
3. Optional TQ+ per-coordinate shift/scale
4. Lloyd-Max scalar quantize (`bits` ∈ {2,3,4}) on Beta((d−1)/2,(d−1)/2)
5. Store RaBitQ-style scale `α = ‖v‖ / ⟨u, x̂⟩`
6. Flat SIMD search over BLOCK=32 FastScan layout (x86: FAISS `PERM0` or vector-major when AVX-512 VNNI is available)

Optional `BucketedTurboQuantIndex`: L2-normalize; assign to nearest **routing** centroid (margin-weighted mean + seam attract); soft energy / hard fission via 1-iter spherical 2-means Voronoi partition; mid-stream densify pulse + recover/late absorb when `expected_n` is set; prepare() applies asymmetric anti-rival push to routing centroids. **Encode** centroids stay frozen at bucket birth / partition; residuals are `x̂ − encode_c`. Query builds FastScan LUTs once on `q̂`, opens the top `n_probe` buckets by routing score, and ranks with `⟨q̂, encode_c⟩ + α · ⟨q̂, r̂⟩`. Defaults match the product fat online champ (`soft309_trig1016`).

`dim` must be a positive multiple of 8, ≤ 16384.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DVECTORCACHE_BUILD_GLOVE=ON \
         -DVECTORCACHE_BUILD_TOOLS=ON \
         -DVECTORCACHE_BUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `VECTORCACHE_BUILD_GLOVE` | ON | ann-benchmarks HDF5 reader (GloVe, SIFT1M) |
| `VECTORCACHE_BUILD_TOOLS` | ON | CLI tools |
| `VECTORCACHE_BUILD_TESTS` | ON | GoogleTest |
| `VECTORCACHE_OPENMP` | ON | OpenMP encode/search |
| `VECTORCACHE_FETCH_DATASETS` | ON | `fetch-datasets` |
| `VECTORCACHE_FETCH_OPENAI` | ON | OpenAI parquet→npy |

## CLI: query-bench

```bash
./query-bench --dataset glove --bits 4 --k 10
./query-bench --dataset glove --bits 4 --calibrate --recall
./query-bench --dataset sift1m --bits 4 --k 10 --limit 0 --query-limit 10000
./query-bench --dataset openai-1536 --bits 2 --k 64
./query-bench --npy data/glove-train-100k.npy --limit 100000 --query-limit 1000
./query-bench --dataset glove --bucketed --n-probe 64 --max-bucket-size 800 --energy-soft 400
```

OpenAI NPY corpora have no HDF5-style `test` split; `--query-split` defaults to `holdout` (last `--query-limit` rows) for them. GloVe and SIFT1M default to `test`.

Reports median batch search latency (`ms_per_query`) and optional **Recall@1@k** / **Recall@k**. Exact top-k IDs are cached under `.cache/exact_topk/` (keyed by dataset/npy, split, index size, query split/limit, and k) and reused on later runs with the same ground-truth parameters.

Bucketed mode flags: `--bucketed`, `--n-probe`, `--max-bucket-size`, `--energy-soft` (plus deprecated `--cos-var-threshold` / `--min-bucket-size` no-ops). `expected_n` is set from the loaded corpus size so the densify pulse schedule runs.

## Library sketch

```cpp
#include "vectorcache/index.hpp"
#include "vectorcache/cluster/kmeans_buckets.hpp"

// Flat
vectorcache::TurboQuantIndex index(/*dim=*/1536, /*bits=*/4);
index.calibrate(sample);   // optional TQ+
index.add(database);       // flat float[n * dim]
index.prepare();
auto res = index.search(queries, /*k=*/10);

// Bucketed (online moving-mean IVF + per-bucket FastScan)
vectorcache::BucketParams p;  // n_probe, max_bucket_size, energy_soft, expected_n, ...
p.expected_n = /*N*/ 0;       // set to corpus size to enable densify pulse schedule
vectorcache::BucketedTurboQuantIndex bindex(/*dim=*/1536, /*bits=*/4, p);
bindex.calibrate(sample);
bindex.add(database);
bindex.prepare();
auto bres = bindex.search(queries, /*k=*/10);
```
