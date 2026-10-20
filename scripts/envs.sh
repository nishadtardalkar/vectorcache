#!/bin/bash
# HPC environment setup for VectorCache (C++).
#
# Override module names if your cluster uses different ones:
#   export VECTORCACHE_MODULE_CURL=libcurl/8.5.0
#   export VECTORCACHE_MODULE_ARROW=apache-arrow/15.0.0
#   export VECTORCACHE_MODULE_HDF5=hdf5/1.14.3
#   export VECTORCACHE_MODULE_OPENBLAS=openblas/0.3.26
#
# Find names on your cluster:
#   module avail 2>&1 | grep -iE 'curl|arrow|hdf5|openblas|mkl'

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

# GCC 14+ libstdc++ provides CXXABI_1.3.15 required by conda-forge / newer Arrow builds.
load_module gcc/14 || load_module gcc/13 || load_module gcc || true
load_module cmake || true

# Capture GCC before later modules (e.g. HDF5) switch PrgEnv to Intel.
_VECTORCACHE_GCC="$(type -P gcc 2>/dev/null || true)"
_VECTORCACHE_GXX="$(type -P g++ 2>/dev/null || true)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_ENV="${VECTORCACHE_ENV_PREFIX:-$PROJECT_ROOT/env}"

if [[ -d "$LOCAL_ENV/lib/cmake/Arrow" || -f "$LOCAL_ENV/lib/libparquet.so" ]]; then
  export CMAKE_PREFIX_PATH="${LOCAL_ENV}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="${LOCAL_ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CURL_MOD="${VECTORCACHE_MODULE_CURL:-curl}"
ARROW_MOD="${VECTORCACHE_MODULE_ARROW:-arrow}"
HDF5_MOD="${VECTORCACHE_MODULE_HDF5:-hdf5}"
OPENBLAS_MOD="${VECTORCACHE_MODULE_OPENBLAS:-openblas}"

load_module "$CURL_MOD" || warn_missing "libcurl for fetch-datasets" "$CURL_MOD"
load_module "$ARROW_MOD" || warn_missing "Apache Arrow for Parquet datasets" "$ARROW_MOD"
load_module "$HDF5_MOD" || warn_missing "HDF5 for GloVe reader" "$HDF5_MOD"
# Landmark projection needs CBLAS (SGEMM). Prefer OpenBLAS; fall back to MKL.
if ! load_module "$OPENBLAS_MOD"; then
  if ! load_module mkl && ! load_module intel-mkl; then
    warn_missing "OpenBLAS/MKL for landmark CBLAS" "$OPENBLAS_MOD"
  fi
fi
# Propagate module prefixes so cmake/VectorcacheBlas.cmake can find cblas.h.
# Modules often only set CPATH / LIBRARY_PATH — derive OPENBLAS_ROOT from those.
_vc_set_openblas_root_from_cblas() {
  local cblas="$1"
  [[ -f "$cblas" ]] || return 1
  local inc root
  inc="$(cd "$(dirname "$cblas")" && pwd)"
  root="$(cd "$inc/.." && pwd)"
  if [[ -f "$root/include/cblas.h" ]]; then
    export OPENBLAS_ROOT="$root"
  else
    export OPENBLAS_ROOT="$inc"
  fi
  return 0
}
if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
  for _ob in /usr /usr/local /opt/OpenBLAS \
      ${OPENBLAS_DIR:-} ${OpenBLAS_DIR:-} ${OPENBLAS_HOME:-}; do
    if [[ -n "$_ob" && -f "$_ob/include/cblas.h" ]]; then
      export OPENBLAS_ROOT="$_ob"
      break
    fi
  done
  unset _ob
fi
if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
  IFS=':' read -r -a _vc_inc_paths <<< "${CPATH:-}:${C_INCLUDE_PATH:-}:${CPLUS_INCLUDE_PATH:-}"
  for _inc in "${_vc_inc_paths[@]}"; do
    [[ -n "$_inc" ]] || continue
    if [[ -f "$_inc/cblas.h" ]]; then
      _vc_set_openblas_root_from_cblas "$_inc/cblas.h" && break
    fi
    if [[ -f "$_inc/openblas/cblas.h" ]]; then
      _vc_set_openblas_root_from_cblas "$_inc/openblas/cblas.h" && break
    fi
  done
  unset _vc_inc_paths _inc
fi
if [[ -z "${OPENBLAS_ROOT:-}" && -n "${LIBRARY_PATH:-}${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -r -a _vc_lib_paths <<< "${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
  for _libdir in "${_vc_lib_paths[@]}"; do
    [[ -n "$_libdir" ]] || continue
    _cand="$(cd "$_libdir/.." 2>/dev/null && pwd)"
    if [[ -n "$_cand" && -f "$_cand/include/cblas.h" ]]; then
      export OPENBLAS_ROOT="$_cand"
      break
    fi
  done
  unset _vc_lib_paths _libdir _cand
fi
if [[ -n "${OPENBLAS_ROOT:-}" ]]; then
  echo "vectorcache: OPENBLAS_ROOT=${OPENBLAS_ROOT}" >&2
fi
unset -f _vc_set_openblas_root_from_cblas

# HDF5 modules often load PrgEnv-intel, but Intel icpx cannot compile libstdc++
# <variant> (used by Arrow) against GCC headers — pin GCC for the whole build.
if [[ -n "$_VECTORCACHE_GCC" && -n "$_VECTORCACHE_GXX" ]]; then
  export CC="${VECTORCACHE_CC:-$_VECTORCACHE_GCC}"
  export CXX="${VECTORCACHE_CXX:-$_VECTORCACHE_GXX}"
fi
unset _VECTORCACHE_GCC _VECTORCACHE_GXX

export VECTORCACHE_DATA_DIR="${VECTORCACHE_DATA_DIR:-./data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 1)}"
