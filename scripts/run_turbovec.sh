#!/usr/bin/env bash
# Clone/build turbovec (Python bindings) and run an official openai/glove suite
# script. Used by `make turbovec` / `make turbovec-setup`.
#
# HPC split (same idea as login/compute):
#   Login (internet):  make turbovec-setup
#   Compute (offline): make turbovec DATASET=openai-1536 BITS=4 ...
#
# Env / make vars:
#   TURBOVEC_DIR       checkout path (default: .ref-turbovec)
#   TURBOVEC_REPO      git URL if clone needed
#   TURBOVEC_DATA_DIR  where suite scripts look (default: ~/data/py-turboquant)
#   DATA_DIR           vectorcache data/ (source of openai-*.npy)
#   DATASET            openai-1536 | openai-3072 | glove  (default: openai-1536)
#   BITS               2 | 4  (default: 4)
#   TURBOVEC_SUITE     speed | recall  (default: speed)
#   TURBOVEC_THREADS   mt | st  (default: mt; speed only)
#   TURBOVEC_ARCH      x86 | arm  (default: auto)
#   FORCE=1            rebuild wheel even if import works
#   SKIP_FAISS=1       for speed suite: time turbovec only (no FAISS)
#   SETUP_ONLY=1       clone + data + bindings; do not run a suite script
#   TURBOVEC_OFFLINE=1 no git/pip network (default for `make turbovec`)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TURBOVEC_DIR="${TURBOVEC_DIR:-$ROOT/.ref-turbovec}"
TURBOVEC_REPO="${TURBOVEC_REPO:-https://github.com/RyanCodrai/turbovec.git}"
TURBOVEC_DATA_DIR="${TURBOVEC_DATA_DIR:-$HOME/data/py-turboquant}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
DATASET="${DATASET:-openai-1536}"
BITS="${BITS:-4}"
TURBOVEC_SUITE="${TURBOVEC_SUITE:-speed}"
TURBOVEC_THREADS="${TURBOVEC_THREADS:-mt}"
FORCE="${FORCE:-}"
SKIP_FAISS="${SKIP_FAISS:-}"
SETUP_ONLY="${SETUP_ONLY:-}"
TURBOVEC_OFFLINE="${TURBOVEC_OFFLINE:-}"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

detect_arch() {
  local m
  m="$(uname -m 2>/dev/null || echo x86_64)"
  case "$m" in
    aarch64|arm64) echo arm ;;
    *) echo x86 ;;
  esac
}
if [[ -z "${TURBOVEC_ARCH:-}" ]]; then
  TURBOVEC_ARCH="$(detect_arch)"
fi

have_checkout() {
  [[ -f "$TURBOVEC_DIR/Cargo.toml" && -d "$TURBOVEC_DIR/turbovec-python" && -d "$TURBOVEC_DIR/benchmarks/suite" ]]
}

dim_tag_for_dataset() {
  case "$1" in
    openai-1536) echo d1536 ;;
    openai-3072) echo d3072 ;;
    glove) echo glove ;;
    *)
      echo "DATASET must be openai-1536, openai-3072, or glove (got: $1)" >&2
      exit 1
      ;;
  esac
}

npy_candidates_for_dataset() {
  case "$1" in
    openai-1536)
      echo "$DATA_DIR/openai-1536.npy"
      echo "$DATA_DIR/openai-1536-101k.npy"
      ;;
    openai-3072)
      echo "$DATA_DIR/openai-3072.npy"
      echo "$DATA_DIR/openai-3072-101k.npy"
      ;;
    glove)
      echo "$DATA_DIR/glove-200-angular.hdf5"
      ;;
  esac
}

find_wheel() {
  ls -t "$TURBOVEC_DIR"/target/wheels/turbovec-*.whl \
        "$TURBOVEC_DIR"/turbovec-python/target/wheels/turbovec-*.whl \
        2>/dev/null | head -1 || true
}

ensure_clone() {
  if have_checkout; then
    echo "turbovec checkout: $TURBOVEC_DIR"
    return
  fi

  # Incomplete / failed clone (e.g. compute-node timeout) — drop and retry or abort.
  if [[ -e "$TURBOVEC_DIR" ]]; then
    echo "Incomplete turbovec checkout at $TURBOVEC_DIR; removing it." >&2
    rm -rf "$TURBOVEC_DIR"
  fi

  if [[ -n "$TURBOVEC_OFFLINE" ]]; then
    echo "Missing turbovec checkout at $TURBOVEC_DIR (offline mode)." >&2
    echo "On a login node with internet, from this same project path, run:" >&2
    echo "  make turbovec-setup" >&2
    echo "Then re-run make turbovec on the compute node." >&2
    exit 1
  fi

  echo "Cloning turbovec into $TURBOVEC_DIR ..."
  git clone --depth 1 "$TURBOVEC_REPO" "$TURBOVEC_DIR"
  if ! have_checkout; then
    echo "Clone finished but checkout looks incomplete: $TURBOVEC_DIR" >&2
    exit 1
  fi
}

link_or_copy() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    # Already present and non-empty — keep it (full 1M corpus preferred over 101k).
    if [[ -s "$dst" ]]; then
      echo "Using existing $dst"
      return
    fi
    rm -f "$dst"
  fi
  if ln -f "$src" "$dst" 2>/dev/null; then
    echo "Hardlinked $dst <- $src"
  elif ln -s "$(cd "$(dirname "$src")" && pwd)/$(basename "$src")" "$dst" 2>/dev/null; then
    echo "Symlinked $dst <- $src"
  else
    cp -f "$src" "$dst"
    echo "Copied $dst <- $src"
  fi
}

ensure_data() {
  mkdir -p "$TURBOVEC_DATA_DIR"
  case "$DATASET" in
    openai-1536|openai-3072)
      local dim="${DATASET#openai-}"
      local dest="$TURBOVEC_DATA_DIR/openai-${dim}.npy"
      if [[ -s "$dest" ]]; then
        echo "Dataset ready: $dest"
        return
      fi
      local cand
      while IFS= read -r cand; do
        [[ -z "$cand" ]] && continue
        if [[ -s "$cand" ]]; then
          link_or_copy "$cand" "$dest"
          return
        fi
      done < <(npy_candidates_for_dataset "$DATASET")
      echo "Missing $DATASET under $DATA_DIR (need openai-${dim}.npy or openai-${dim}-101k.npy)." >&2
      echo "Run: make login DATASETS=$DATASET" >&2
      exit 1
      ;;
    glove)
      local dest="$TURBOVEC_DATA_DIR/glove-200-angular.hdf5"
      if [[ -s "$dest" ]]; then
        echo "Dataset ready: $dest"
        return
      fi
      local src="$DATA_DIR/glove-200-angular.hdf5"
      if [[ -s "$src" ]]; then
        link_or_copy "$src" "$dest"
        return
      fi
      echo "Missing glove under $DATA_DIR. Run: make login DATASETS=glove" >&2
      exit 1
      ;;
  esac
}

install_wheel() {
  local wheel="$1"
  echo "Installing $wheel ..."
  if [[ -n "$TURBOVEC_OFFLINE" ]]; then
    "$PYTHON" -m pip install --force-reinstall --no-deps --no-index "$wheel"
  else
    "$PYTHON" -m pip install --force-reinstall --no-deps "$wheel"
  fi
  "$PYTHON" -c "from turbovec import TurboQuantIndex; print('turbovec ready')"
}

ensure_bindings() {
  if [[ -z "$FORCE" ]] && "$PYTHON" -c "from turbovec import TurboQuantIndex" >/dev/null 2>&1; then
    echo "turbovec Python import OK ($PYTHON)"
    return
  fi

  local wheel
  wheel="$(find_wheel)"

  if [[ -n "$TURBOVEC_OFFLINE" ]]; then
    if [[ -n "$wheel" && -f "$wheel" ]]; then
      install_wheel "$wheel"
      return
    fi
    echo "turbovec is not importable and no local wheel was found under $TURBOVEC_DIR (offline)." >&2
    echo "On a login node with internet, using the same Python env, run:" >&2
    echo "  make turbovec-setup" >&2
    exit 1
  fi

  if ! "$PYTHON" -c "import maturin" >/dev/null 2>&1; then
    echo "Installing maturin ..."
    "$PYTHON" -m pip install -U 'maturin>=1.12,<2.0'
  fi
  echo "Building turbovec Python wheel (release) ..."
  (
    cd "$TURBOVEC_DIR/turbovec-python"
    "$PYTHON" -m maturin build --release
  )
  wheel="$(find_wheel)"
  if [[ -z "$wheel" || ! -f "$wheel" ]]; then
    echo "maturin build produced no turbovec wheel under $TURBOVEC_DIR" >&2
    exit 1
  fi
  install_wheel "$wheel"
}

suite_script() {
  local tag
  tag="$(dim_tag_for_dataset "$DATASET")"
  case "$TURBOVEC_SUITE" in
    recall)
      if [[ "$tag" == glove ]]; then
        echo "$TURBOVEC_DIR/benchmarks/suite/recall_glove_${BITS}bit.py"
      else
        echo "$TURBOVEC_DIR/benchmarks/suite/recall_${tag}_${BITS}bit.py"
      fi
      ;;
    speed)
      if [[ "$tag" == glove ]]; then
        echo "speed suite has no glove scripts; use TURBOVEC_SUITE=recall" >&2
        exit 1
      fi
      echo "$TURBOVEC_DIR/benchmarks/suite/speed_${tag}_${BITS}bit_${TURBOVEC_ARCH}_${TURBOVEC_THREADS}.py"
      ;;
    *)
      echo "TURBOVEC_SUITE must be speed or recall (got: $TURBOVEC_SUITE)" >&2
      exit 1
      ;;
  esac
}

run_speed_tq_only() {
  # Same timing protocol as suite/speed_*.py but skip FAISS (faster local loop).
  local dim="${DATASET#openai-}"
  "$PYTHON" - "$dim" "$BITS" "$TURBOVEC_DATA_DIR" <<'PY'
import os, sys, time, json
import numpy as np
from turbovec import TurboQuantIndex

dim, bits, data_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
path = os.path.join(data_dir, f"openai-{dim}.npy")
all_vecs = np.load(path)
rng = np.random.RandomState(42)
idx = rng.permutation(len(all_vecs))
database = all_vecs[idx[:100_000]].astype(np.float32, copy=False)
queries = all_vecs[idx[100_000:101_000]].astype(np.float32, copy=False)
database /= np.linalg.norm(database, axis=-1, keepdims=True)
queries /= np.linalg.norm(queries, axis=-1, keepdims=True)

tq = TurboQuantIndex(dim=dim, bit_width=bits)
tq.add(database)
tq.search(queries[:1], k=64)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    tq.search(queries, k=64)
    times.append((time.perf_counter() - t0) / len(queries) * 1000)
ms = sorted(times)[2]
out = {
    "dim": dim,
    "bit_width": bits,
    "n": int(len(database)),
    "nq": int(len(queries)),
    "k": 64,
    "tq_ms_per_query": round(ms, 6),
    "runs_ms_per_query": [round(t, 6) for t in times],
}
print(json.dumps(out, indent=2))
PY
}

main() {
  if [[ "${1:-}" == "--setup-only" ]]; then
    SETUP_ONLY=1
  fi
  if [[ "$BITS" != "2" && "$BITS" != "4" ]]; then
    echo "BITS must be 2 or 4 (got: $BITS)" >&2
    exit 1
  fi

  # Setup is online. Bench defaults offline (compute nodes have no GitHub).
  if [[ -n "$SETUP_ONLY" ]]; then
    TURBOVEC_OFFLINE=""
  elif [[ "${TURBOVEC_ALLOW_NET:-}" == "1" ]]; then
    TURBOVEC_OFFLINE=""
  elif [[ -z "${TURBOVEC_OFFLINE}" ]]; then
    TURBOVEC_OFFLINE=1
  fi

  ensure_clone
  ensure_data
  ensure_bindings

  if [[ -n "$SETUP_ONLY" ]]; then
    echo "turbovec setup complete (DATASET=$DATASET)."
    echo "On the compute node, run: make turbovec DATASET=$DATASET BITS=$BITS ..."
    return
  fi

  echo "=== turbovec bench: DATASET=$DATASET BITS=$BITS SUITE=$TURBOVEC_SUITE ARCH=$TURBOVEC_ARCH THREADS=$TURBOVEC_THREADS ==="

  if [[ "$TURBOVEC_SUITE" == speed && -n "$SKIP_FAISS" ]]; then
    run_speed_tq_only
    return
  fi

  local script
  script="$(suite_script)"
  if [[ ! -f "$script" ]]; then
    echo "Suite script not found: $script" >&2
    exit 1
  fi
  echo "Running $script"
  "$PYTHON" "$script"
}

main "$@"
