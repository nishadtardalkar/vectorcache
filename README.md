# VectorCache

Block-based multi-level quantized vector retrieval engine (C++20).

VectorCache implements the ingestion pipeline for approximate nearest neighbor search: SRHT rotation, L1 4D-to-1bit quantization, and 1024-vector block storage.

## Requirements

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.20+
- For CLI tools (`fetch-datasets`):
  - libcurl
  - Apache Arrow C++ with Parquet
- For GloVe HDF5 support (`-DVECTORCACHE_BUILD_GLOVE=ON`):
  - HDF5 C library

On HPC clusters, load modules before building:

```bash
source scripts/envs.sh
# Or manually (names vary by site):
module load gcc cmake
module load curl arrow hdf5
```

If you see `Unable to locate a modulefile for 'curl'` or `'arrow'`, those exact names are not on your cluster. Search for the real names:

```bash
module avail 2>&1 | grep -iE 'curl|arrow|hdf5'
```

Then either edit `scripts/envs.sh` or export overrides before `make login`:

```bash
export VECTORCACHE_MODULE_CURL=libcurl/8.5.0      # example
export VECTORCACHE_MODULE_ARROW=apache-arrow/15.0.0
export VECTORCACHE_MODULE_HDF5=hdf5/1.14.3
make login
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

### SRHT round count (compile-time)

Default is 3 rounds (`H·D₃·H·D₂·H·D₁`). To build with a single `H·D` round:

```bash
cmake .. -DVECTORCACHE_SRHT_ROUNDS=1
```

Allowed values: `1`, `2`, or `3`. Reconfigure and rebuild after changing; there is no runtime flag.

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

### Link error: `__cxa_call_terminate@CXXABI_1.3.15`

This means **libparquet was built with a newer GCC/libstdc++ than your linker is using** (common with conda/mamba Arrow in `env/` on clusters that default to GCC 12/13).

Fix options (pick one):

1. **Load GCC 14+** before building (if your cluster has it):
   ```bash
   module avail 2>&1 | grep -i gcc
   module load gcc/14    # example
   make clean && make login
   ```

2. **Use a local conda/mamba env** for Arrow and let CMake link its libstdc++ (automatic if `env/` exists and you reconfigure):
   ```bash
   micromamba create -p ./env -c conda-forge "arrow>=15" parquet
   source scripts/envs.sh
   make clean && make login
   ```

3. **Use the cluster Arrow module** built with the same GCC you compile with, instead of conda Arrow:
   ```bash
   export VECTORCACHE_MODULE_ARROW=apache-arrow/15.0.0   # example
   unset CMAKE_PREFIX_PATH   # drop ./env if set
   make clean && make login
   ```

4. **Skip OpenAI datasets** if you only need GloVe:
   ```bash
   make login DATASETS=glove
   ```

## Missing dependencies on HPC

Apache Arrow is only required when downloading OpenAI datasets (`openai-1536`, `openai-3072`, or `DATASETS=all`). GloVe-only login does not need Arrow:

```bash
make login DATASETS=glove
```

For all datasets, Arrow must be installed separately and pointed at via CMake (not bundled in this repo):

```bash
export CMAKE_PREFIX_PATH=/path/to/arrow/prefix   # contains lib/cmake/Arrow/ArrowConfig.cmake
make login
```

Or set `Arrow_DIR` directly:

```bash
make login CMAKE_OPTS="-DArrow_DIR=/path/to/lib/cmake/Arrow"
```

If Arrow or HDF5 are unavailable on your cluster:

1. Request them from your cluster admin, or
2. Copy pre-downloaded `data/` from a machine that already has the datasets, or
3. Use GloVe only: `make login DATASETS=glove` / `make compute DATASET=glove`

To build without dataset fetching (library + tests only):

```bash
cmake .. -DVECTORCACHE_BUILD_TOOLS=OFF
```

## License

MIT OR Apache-2.0
