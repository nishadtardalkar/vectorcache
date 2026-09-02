#!/bin/bash
# HPC environment setup for VectorCache (C++).
#
# Override module names if your cluster uses different ones:
#   export VECTORCACHE_MODULE_CURL=libcurl/8.5.0
#   export VECTORCACHE_MODULE_ARROW=apache-arrow/15.0.0
#   export VECTORCACHE_MODULE_HDF5=hdf5/1.14.3
#
# Find names on your cluster:
#   module avail 2>&1 | grep -iE 'curl|arrow|hdf5'

module_exists() {
  local name="$1"
  [[ -z "$name" ]] && return 1
  type module &>/dev/null || return 1
  # Lmod: silent availability check (avoids "Unable to locate modulefile" noise)
  if module is-avail "$name" &>/dev/null; then
    return 0
  fi
  # Environment Modules (Tcl): list modules and match name or versioned suffix
  module -t avail 2>&1 | grep -E "(^|/)$name(/|$)" -q
}

load_module() {
  local name="$1"
  if module_exists "$name"; then
    module load "$name"
    return 0
  fi
  return 1
}

warn_missing() {
  local label="$1"
  local tried="$2"
  echo "WARNING: no module named '$tried' ($label). Search with: module avail 2>&1 | grep -i ${tried%%/*}" >&2
}

load_module gcc || true
load_module cmake || true

CURL_MOD="${VECTORCACHE_MODULE_CURL:-curl}"
ARROW_MOD="${VECTORCACHE_MODULE_ARROW:-arrow}"
HDF5_MOD="${VECTORCACHE_MODULE_HDF5:-hdf5}"

load_module "$CURL_MOD" || warn_missing "libcurl for fetch-datasets" "$CURL_MOD"
load_module "$ARROW_MOD" || warn_missing "Apache Arrow for Parquet datasets" "$ARROW_MOD"
load_module "$HDF5_MOD" || warn_missing "HDF5 for GloVe reader" "$HDF5_MOD"

export VECTORCACHE_DATA_DIR="${VECTORCACHE_DATA_DIR:-./data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 1)}"
