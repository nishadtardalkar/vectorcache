# VectorCache

Flat TurboQuant ANN engine (C++20) matching [turbovec](https://github.com/RyanCodrai/turbovec)'s core path: ChaCha8 block-Hadamard rotation, Lloyd-Max Beta codebook, bit-plane encode, FastScan blocked layout, and SIMD search — plus dataset fetch helpers for TurboVec/TurboQuant benchmarks.

## Requirements

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.20+
- OpenMP (recommended)
- For `fetch-datasets`: libcurl; Apache Arrow/Parquet for OpenAI datasets
- For GloVe: HDF5 C library

On HPC clusters:

```bash
source scripts/envs.sh
```

## HPC workflow

```bash
# Login node (internet):
make login
make login DATASETS=glove

# Compute node (offline):
make compute
make compute BITS=4 RECALL=1 CALIBRATE=1
make compute DATASET=openai-1536 BITS=2 K=64
```

`make login` configures CMake and downloads datasets into `data/`.  
`make compute` reconfigures offline, builds tests + `query-bench`, runs `ctest`, then runs `query-bench`.

## Algorithm (turbovec-compatible)

1. L2-normalize
2. K=2 rounds of global ChaCha8 Fisher–Yates → ±1 signs → normalized block Walsh–Hadamard (`B = dim & -dim`)
3. Optional TQ+ per-coordinate shift/scale
4. Lloyd-Max scalar quantize (`bits` ∈ {2,3,4}) on Beta((d−1)/2,(d−1)/2)
5. Store RaBitQ-style scale `α = ‖v‖ / ⟨u, x̂⟩`
6. Flat SIMD search over BLOCK=32 FastScan layout (x86: FAISS `PERM0` or vector-major when AVX-512 VNNI is available)

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
| `VECTORCACHE_BUILD_GLOVE` | ON | GloVe HDF5 reader |
| `VECTORCACHE_BUILD_TOOLS` | ON | CLI tools |
| `VECTORCACHE_BUILD_TESTS` | ON | GoogleTest |
| `VECTORCACHE_OPENMP` | ON | OpenMP encode/search |
| `VECTORCACHE_FETCH_DATASETS` | ON | `fetch-datasets` |
| `VECTORCACHE_FETCH_OPENAI` | ON | OpenAI parquet→npy |

## CLI: query-bench

```bash
./query-bench --dataset glove --bits 4 --k 10
./query-bench --dataset glove --bits 4 --calibrate --recall
./query-bench --npy data/glove-train-100k.npy --limit 100000 --query-limit 1000
```

Reports median batch search latency (`ms_per_query`) and optional **Recall@1@k** / **Recall@k**.

## Library sketch

```cpp
#include "vectorcache/index.hpp"

vectorcache::TurboQuantIndex index(/*dim=*/1536, /*bits=*/4);
index.calibrate(sample);   // optional TQ+
index.add(database);       // flat float[n * dim]
index.prepare();
auto res = index.search(queries, /*k=*/10);
```
