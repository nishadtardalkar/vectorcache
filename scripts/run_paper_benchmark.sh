#!/usr/bin/env bash
# Ours-only paper experiment pack → one results file.
#
# Sections:
#   A  ANN-only            (FLOAT_RESCORE=0, m=1024, bits=4, k=64)
#   B  ANN + float re-rank (FLOAT_RESCORE=1, same knobs)
#   C  ablations: m∈{256,512}, bits=2, planes-off
#   D  scraped key=value summary (latency, dual recall, bytes/vector)
#
# Protocol: openai-1536, n=100K, nq=1K holdout (last rows), OMP=8, RECALL=1.
# This harness does not invoke turbovec or FAISS.
#
# Usage (via Make):  make benchmark
# Env overrides: DATASET BITS LIMIT QUERY_LIMIT K LANDMARKS OMP_NUM_THREADS
#                SKIP_ABLATIONS=1  BENCHMARK_OUT=path  BUILD_DIR=build

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/envs.sh"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_DIR_ABS="$(cd "$BUILD_DIR" 2>/dev/null && pwd || echo "$ROOT/$BUILD_DIR")"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
DATA_DIR="${DATA_DIR:-data}"
DATASET="${DATASET:-openai-1536}"
LIMIT="${LIMIT:-100000}"
QUERY_LIMIT="${QUERY_LIMIT:-1000}"
BITS="${BITS:-4}"
K="${K:-64}"
LANDMARKS="${LANDMARKS:-1024}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
SKIP_ABLATIONS="${SKIP_ABLATIONS:-}"
LOGIN_READY="${BUILD_DIR}/.login-ready"

STAMP="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || date +%Y%m%d)"
OUT_DIR="${BENCHMARK_OUT_DIR:-results}"
OUT="${BENCHMARK_OUT:-$OUT_DIR/paper_benchmark_${DATASET}_${STAMP}.txt}"
mkdir -p "$(dirname "$OUT")"

export OMP_NUM_THREADS

log() { printf '%s\n' "$*" | tee -a "$OUT"; }
hr() { log "------------------------------------------------------------------------"; }

section() {
  log ""
  hr
  log "## $1"
  hr
  log ""
}

run_logged() {
  log "+ $*"
  set +e
  "$@" 2>&1 | tee -a "$OUT"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "!! FAILED (exit $rc): $*"
  fi
  return 0
}

find_query_bench() {
  local candidate
  for candidate in \
    "$BUILD_DIR_ABS/query-bench" \
    "$BUILD_DIR_ABS/$BUILD_TYPE/query-bench"; do
    if [[ -x "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  find "$BUILD_DIR_ABS" -maxdepth 3 \
    \( -name 'query-bench' -o -name 'query-bench.exe' \) -type f -print -quit 2>/dev/null || true
}

# --- preamble -----------------------------------------------------------------
: >"$OUT"
log "# VectorCache paper benchmark pack (ours only)"
log "generated_utc=$STAMP"
log "host=$(hostname 2>/dev/null || echo unknown)"
log "cwd=$ROOT"
if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git -C "$ROOT" status -sb 2>/dev/null | head -1))"
fi
log "dataset=$DATASET limit=$LIMIT query_limit=$QUERY_LIMIT bits=$BITS k=$K landmarks_default=$LANDMARKS"
log "OMP_NUM_THREADS=$OMP_NUM_THREADS SKIP_ABLATIONS=${SKIP_ABLATIONS:-0}"
log "output=$OUT"
log ""
log "Metrics:"
log "  ms_per_query / ann_ms_per_query / rerank_ms_per_query  (median of timing-runs)"
log "  recall1@t = 1-recall (exact NN in approx[:t])"
log "  recall@t  = set-overlap |approx[:t] ∩ exact[:t]| / t"
log "  codes_bytes_per_vector / index_bytes_per_vector / float32_bytes_per_vector"
log ""
log "Protocol: contiguous holdout (last query_limit rows after load of limit+query_limit)."

# --- build once ---------------------------------------------------------------
section "0. Build (query-bench + tests)"

if [[ ! -f "$LOGIN_READY" ]]; then
  log "ERROR: missing $LOGIN_READY — run 'make login' on a login node first."
  exit 1
fi

CMAKE_COMPILER_FLAGS=(
  -DCMAKE_C_COMPILER="${CC:-gcc}"
  -DCMAKE_CXX_COMPILER="${CXX:-g++}"
)
CMAKE_COMPUTE_FLAGS=(
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DVECTORCACHE_BUILD_GLOVE=ON
  -DVECTORCACHE_BUILD_TOOLS=ON
  -DVECTORCACHE_BUILD_TESTS=ON
  -DVECTORCACHE_FETCH_DATASETS=OFF
  -DVECTORCACHE_FETCH_OPENAI=OFF
)

log "+ cmake configure"
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  "${CMAKE_COMPILER_FLAGS[@]}" \
  "${CMAKE_COMPUTE_FLAGS[@]}" 2>&1 | tee -a "$OUT"
log "+ cmake --build query-bench"
cmake --build "$BUILD_DIR" --target vectorcache_tests query-bench -j"$JOBS" 2>&1 | tee -a "$OUT"
log "+ ctest"
ctest --test-dir "$BUILD_DIR" --output-on-failure 2>&1 | tee -a "$OUT"

QUERY_BENCH="$(find_query_bench)"
if [[ -z "$QUERY_BENCH" || ! -f "$QUERY_BENCH" ]]; then
  log "ERROR: query-bench not found under $BUILD_DIR_ABS"
  exit 1
fi
log "query-bench=$QUERY_BENCH"

ours() {
  local label="$1"
  shift
  local landmarks="${LANDMARKS}"
  local bits="${BITS}"
  local float_rescore=1
  local planes_off=0
  local extra=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --landmarks) landmarks="$2"; shift 2 ;;
      --bits) bits="$2"; shift 2 ;;
      --no-float-rescore) float_rescore=0; shift ;;
      --float-rescore) float_rescore=1; shift ;;
      --planes-off) planes_off=1; shift ;;
      *) extra+=("$1"); shift ;;
    esac
  done

  local args=(
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
    --limit "$LIMIT"
    --query-limit "$QUERY_LIMIT"
    --bits "$bits"
    --landmarks "$landmarks"
    --k "$K"
    --recall
  )
  if [[ "$float_rescore" -eq 0 ]]; then
    args+=(--no-float-rescore)
  else
    args+=(--float-rescore)
  fi
  args+=("${extra[@]}")

  log ""
  log "### $label"
  log "config landmarks=$landmarks bits=$bits k=$K float_rescore=$float_rescore planes_off=$planes_off"
  if [[ "$planes_off" -eq 1 ]]; then
    run_logged env TURBOVEC_4BIT_PLANES=0 "$QUERY_BENCH" "${args[@]}"
  else
    run_logged "$QUERY_BENCH" "${args[@]}"
  fi
}

# --- A / B primary cells ------------------------------------------------------
section "A. ANN-only (FLOAT_RESCORE=0) — latency + dual recall"
ours "A primary m=${LANDMARKS} bits=${BITS}" --landmarks "$LANDMARKS" --bits "$BITS" --no-float-rescore

section "B. ANN + float re-rank (FLOAT_RESCORE=1) — end-to-end story"
ours "B primary m=${LANDMARKS} bits=${BITS}" --landmarks "$LANDMARKS" --bits "$BITS" --float-rescore

# --- C ablations --------------------------------------------------------------
section "C. Ablations (m, bits, planes) — dual recall + split timing"

if [[ -n "$SKIP_ABLATIONS" ]]; then
  log "SKIP_ABLATIONS=1 — skipping landmark/bits/planes sweeps."
else
  for m in 256 512; do
    ours "C.m=${m} float_rescore=0" --landmarks "$m" --bits "$BITS" --no-float-rescore
    ours "C.m=${m} float_rescore=1" --landmarks "$m" --bits "$BITS" --float-rescore
  done

  ours "C.bits=2 float_rescore=0" --landmarks "$LANDMARKS" --bits 2 --no-float-rescore
  ours "C.bits=2 float_rescore=1" --landmarks "$LANDMARKS" --bits 2 --float-rescore

  ours "C.planes=off float_rescore=0" --landmarks "$LANDMARKS" --bits "$BITS" --no-float-rescore --planes-off
  ours "C.planes=off float_rescore=1" --landmarks "$LANDMARKS" --bits "$BITS" --float-rescore --planes-off
fi

# --- D scrape -----------------------------------------------------------------
section "D. Scraped summary (machine-parseable lines)"

log "# key=value lines from this file matching paper columns"
if command -v awk >/dev/null 2>&1; then
  awk '
    /^### / { label=$0; next }
    /^config / { cfg=$0; next }
    /^ingest_ms=/ || /^codes_bytes=/ || /^search_median_ms=/ || /^recall1@/ || /^recall@/ {
      if (label != "") { print label; label="" }
      if (cfg != "") { print cfg; cfg="" }
      print
    }
  ' "$OUT" >"$OUT.tmp_scrape" || true
  if [[ -f "$OUT.tmp_scrape" ]]; then
    log "----- begin scrape -----"
    cat "$OUT.tmp_scrape" >>"$OUT" || true
    log "----- end scrape -----"
    rm -f "$OUT.tmp_scrape"
  fi
else
  log "(awk unavailable — open the sections above for numbers)"
fi

log ""
hr
log "DONE. Results: $OUT"
hr
printf '%s\n' "$OUT"
