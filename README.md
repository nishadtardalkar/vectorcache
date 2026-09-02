# VectorCache

Block-based multi-level quantized vector retrieval engine (C++20).

VectorCache implements the ingestion pipeline for approximate nearest neighbor search: SRHT rotation, L1 4D-to-1bit quantization, and 1024-vector block storage.

## Requirements

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.20+
- OpenMP
- For CLI tools (`fetch-datasets`):
  - libcurl
  - Apache Arrow C++ with Parquet
- For GloVe HDF5 support (`-DVECTORCACHE_BUILD_GLOVE=ON`):
  - HDF5 C library

On HPC clusters, load modules before building:

```bash
source scripts/envs.sh
# Or manually:
module load gcc cmake
module load curl arrow hdf5
```

### HPC workflow

Use the root `Makefile` to split internet-dependent work (login node) from offline build/test/benchmark (compute node). Both nodes must see the same project path.

```bash
# On login node (internet):
make login

# On compute node (no internet):
make compute DATASET=glove

# Optional benchmark caps/extras:
make compute DATASET=glove BENCH_EXTRA_ARGS="--limit 50000"
```

`make login` runs CMake configure (FetchContent clones), builds `fetch-datasets`, and downloads datasets into `data/`. `make compute` reconfigures with `FETCHCONTENT_FULLY_DISCONNECTED=ON`, builds everything, runs `ctest`, and runs `ingest-bench` against the required `DATASET` (e.g. `glove`, `openai-1536`, `openai-3072`).

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
| `VECTORCACHE_BUILD_GLOVE` | ON | Enable GloVe HDF5 reader |
| `VECTORCACHE_BUILD_TOOLS` | ON | Build CLI tools (requires curl + Arrow) |
| `VECTORCACHE_BUILD_TESTS` | ON | Build GoogleTest suite |

### SIMD

The library requires AVX-512F/DQ/BW. GCC/Clang builds use `-mavx512f -mavx512dq -mavx512bw -mfma`; MSVC uses `/arch:AVX512`. There are no scalar fallbacks.

For maximum single-node performance on homogeneous clusters:

```bash
cmake .. -DCMAKE_CXX_FLAGS="-march=native"
```

### OpenMP

Set thread count at runtime:

```bash
export OMP_NUM_THREADS=32
```

## CLI tools

### fetch-datasets

Download benchmark datasets into `data/`:

```bash
./fetch-datasets all
./fetch-datasets glove openai-1536
./fetch-datasets --data-dir data --force openai-3072
```

### ingest-sample

Ingest vectors and optionally report variance:

```bash
./ingest-sample --npy data/.cache/glove-sample-100.npy
./ingest-sample --dataset glove --limit 100 --variance
./ingest-sample --dataset glove --limit 100 --variance --show-index 0
```

Environment variables:
- `VECTORCACHE_DATASET` (default: `glove`)
- `VECTORCACHE_DATA_DIR` (default: `data`)

### ingest-bench

Profile ingestion stage hot paths:

```bash
./ingest-bench --dataset glove
./ingest-bench --dataset glove --limit 50000
./ingest-bench --npy data/openai-1536.npy --limit 50000
```

## Project layout

```
include/vectorcache/   Public headers
src/                   Library implementation
tools/                 CLI executables
tests/                 GoogleTest suite
data/                  Dataset storage (gitignored)
scripts/envs.sh        HPC module setup
```

## Datasets

| Dataset | Dim | Padded | Format |
|---------|-----|--------|--------|
| GloVe | 200 | 256 | HDF5 |
| OpenAI-1536 | 1536 | 2048 | NPY |
| OpenAI-3072 | 3072 | 4096 | NPY |

## Missing dependencies on HPC

If Arrow or HDF5 modules are unavailable:

1. Request them from your cluster admin, or
2. Install via Spack: `spack install arrow +parquet hdf5 curl`, then
   `cmake .. -DCMAKE_PREFIX_PATH=$(spack location -i arrow)`

To build without dataset fetching (library + tests only):

```bash
cmake .. -DVECTORCACHE_BUILD_TOOLS=OFF
```

## License

MIT OR Apache-2.0
