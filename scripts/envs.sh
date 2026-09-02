#!/bin/bash
# HPC environment setup for VectorCache (C++).
# Adjust module names to match your cluster.

module load gcc cmake 2>/dev/null || true
module load curl 2>/dev/null || true
module load arrow 2>/dev/null || true
module load hdf5 2>/dev/null || true

export VECTORCACHE_DATA_DIR="${VECTORCACHE_DATA_DIR:-./data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 1)}"
