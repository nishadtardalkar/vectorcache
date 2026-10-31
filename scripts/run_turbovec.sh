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
#   BITS               2 | 4  (default: 4 — published staged cell)
#   TURBOVEC_SUITE     speed | recall  (default: speed)
#   TURBOVEC_THREADS   mt | st  (default: mt; speed only)
#   TURBOVEC_ARCH      x86 | arm  (default: auto)
#   TURBOVEC_RAYON_THREADS  Rayon pool size for mt (default: 8 — matches
#                      published c3/c4a-standard-8 ~0.12 ms/query cell)
#   TURBOVEC_FULLSCAN=1  force whole-index 4-bit scan (TURBOVEC_4BIT_PLANES=0);
#                      default is staged planes (sign → shortlist → exact)
#   FORCE=1            rebuild wheel even if import works
#   SKIP_FAISS=1       for speed suite: time turbovec only (no FAISS)
#   TURBOVEC_STOCK=1   run upstream benchmarks/suite/*.py as-is (may OOM on full npy)
#   SETUP_ONLY=1       clone + data + bindings; do not run a suite script
#   TURBOVEC_OFFLINE=1 no git/pip network (default for `make turbovec`)
#
# turbovec-setup bootstraps from a clean machine: rustup (if no cargo),
# maturin, numpy, and faiss-cpu (unless SKIP_FAISS=1).
#
# Default `make turbovec` reproduces the published openai-1536 4-bit staged
# multi-thread speed cell (~0.12 ms/query on GCP c3/c4a-standard-8): 100k+1k,
# seed=42, k=64, median of 5, RAYON_NUM_THREADS=8, staged planes on. The mmap
# harness matches that protocol — stock scripts np.load the whole corpus then
# train FAISS PQFastScan(m=dim), which often gets SIGKILL/OOM.

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
# Published mt cell is an 8-vCPU box; pin Rayon so big HPC nodes stay comparable.
TURBOVEC_RAYON_THREADS="${TURBOVEC_RAYON_THREADS:-8}"
TURBOVEC_FULLSCAN="${TURBOVEC_FULLSCAN:-}"
FORCE="${FORCE:-}"
SKIP_FAISS="${SKIP_FAISS:-}"
TURBOVEC_STOCK="${TURBOVEC_STOCK:-}"
SETUP_ONLY="${SETUP_ONLY:-}"
TURBOVEC_OFFLINE="${TURBOVEC_OFFLINE:-}"
# Rust + cargo target dir must live on an exec-capable filesystem. Shared project
# disks (e.g. /rsstu) are often mounted noexec — default to $HOME, not $ROOT.
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/vectorcache-turbovec-target}"
export CARGO_HOME RUSTUP_HOME CARGO_TARGET_DIR

# Prefer an activated venv/conda interpreter. Bare `python3` often resolves to
# /usr/bin/python3 (no pip) even when env/bin/python is on PATH first under `python`.
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    command -v "$PYTHON" >/dev/null 2>&1 && { command -v "$PYTHON"; return; }
    [[ -x "$PYTHON" ]] && { echo "$PYTHON"; return; }
  fi
  local cand
  for cand in \
      "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
      "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
      python \
      python3; do
    [[ -z "$cand" ]] && continue
    if [[ -x "$cand" ]]; then
      echo "$cand"
      return
    fi
    if command -v "$cand" >/dev/null 2>&1; then
      command -v "$cand"
      return
    fi
  done
  return 1
}
PYTHON="$(pick_python || true)"
if [[ -z "$PYTHON" ]]; then
  echo "No Python interpreter found. Activate your env or set PYTHON=/path/to/python." >&2
  exit 1
fi

source_cargo_env() {
  if [[ -f "$CARGO_HOME/env" ]]; then
    # shellcheck disable=SC1090
    source "$CARGO_HOME/env"
  fi
  export PATH="$CARGO_HOME/bin:${PATH:-}"
}

rust_works() {
  source_cargo_env
  command -v rustc >/dev/null 2>&1 || return 1
  command -v cargo >/dev/null 2>&1 || return 1
  rustc -vV >/dev/null 2>&1 || return 1
  cargo -V >/dev/null 2>&1 || return 1
  return 0
}

warn_if_noexec_dir() {
  local dir="$1"
  local probe
  mkdir -p "$dir" 2>/dev/null || return 0
  probe="$(mktemp -p "$dir" .execprobe.XXXXXX 2>/dev/null || true)"
  [[ -n "$probe" ]] || return 0
  printf '#!/bin/sh\nexit 0\n' >"$probe"
  chmod +x "$probe" || true
  if ! "$probe" >/dev/null 2>&1; then
    echo "WARNING: $dir appears noexec (cannot run binaries here)." >&2
    echo "         Rust will be installed under \$HOME instead." >&2
    rm -f "$probe"
    return 1
  fi
  rm -f "$probe"
  return 0
}

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
  ls -t "$CARGO_TARGET_DIR"/wheels/turbovec-*.whl \
        "$TURBOVEC_DIR"/target/wheels/turbovec-*.whl \
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

pip_install() {
  "$PYTHON" -m pip install "$@"
}

ensure_python() {
  if [[ ! -x "$PYTHON" ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python not found (PYTHON=$PYTHON). Activate your env or set PYTHON=..." >&2
    exit 1
  fi
  if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "pip missing for $PYTHON." >&2
    echo "  sys.executable=$("$PYTHON" -c 'import sys; print(sys.executable)' 2>/dev/null || echo '?')" >&2
    if [[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
      echo "  Active env detected but this interpreter has no pip — try: python -m ensurepip --upgrade" >&2
    else
      echo "  No venv/conda is active. Activate ./env first, or: PYTHON=./env/bin/python make turbovec-setup" >&2
    fi
    exit 1
  fi
  echo "Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
}

ensure_python_packages() {
  ensure_python
  if [[ -n "$TURBOVEC_OFFLINE" ]]; then
    local missing=()
    "$PYTHON" -c "import numpy" >/dev/null 2>&1 || missing+=(numpy)
    if [[ -z "$SKIP_FAISS" ]]; then
      "$PYTHON" -c "import faiss" >/dev/null 2>&1 || missing+=(faiss)
    fi
    if ((${#missing[@]})); then
      echo "Missing Python packages offline: ${missing[*]}" >&2
      echo "On a login node with the same env, run: make turbovec-setup" >&2
      exit 1
    fi
    return
  fi

  echo "Ensuring numpy / maturin (and faiss-cpu unless SKIP_FAISS=1) ..."
  pip_install -U pip setuptools wheel
  pip_install -U 'numpy>=1.20' 'maturin>=1.12,<2.0'
  if [[ -z "$SKIP_FAISS" ]]; then
    if ! "$PYTHON" -c "import faiss" >/dev/null 2>&1; then
      # Official suite scripts `import faiss`. Prefer faiss-cpu; fall back to conda hint.
      if ! pip_install -U faiss-cpu; then
        echo "WARNING: faiss-cpu pip install failed. Official speed/recall scripts need faiss." >&2
        echo "  conda install -c pytorch faiss-cpu   # or set SKIP_FAISS=1 for TQ-only speed" >&2
      fi
    fi
  fi
}

try_module_rust() {
  type module >/dev/null 2>&1 || return 1
  local m
  for m in rust rust/1.80 rust/1.78 rust/1.75; do
    if module is-avail "$m" >/dev/null 2>&1 || module -t avail 2>&1 | grep -Eq "(^|/)${m}(/|$)"; then
      module load "$m" 2>/dev/null || continue
      if command -v cargo >/dev/null 2>&1; then
        echo "Loaded module $m ($(cargo --version))"
        return 0
      fi
    fi
  done
  return 1
}

ensure_rust() {
  # Drop a previous project-local install that landed on a noexec mount.
  local stale
  for stale in "$ROOT/.ref-turbovec-cargo" "$ROOT/.ref-turbovec-rustup"; do
    if [[ -e "$stale" ]]; then
      echo "Removing stale noexec Rust install: $stale"
      rm -rf "$stale"
    fi
  done

  if rust_works; then
    echo "Rust OK: $(rustc --version 2>/dev/null); $(cargo --version 2>/dev/null)"
    return
  fi
  if try_module_rust && rust_works; then
    return
  fi
  if [[ -n "$TURBOVEC_OFFLINE" ]]; then
    echo "cargo/rustc not runnable (offline). Run make turbovec-setup on a login node first." >&2
    exit 1
  fi

  # Prefer $HOME; fall back only if caller overrode CARGO_HOME to an exec fs.
  if ! warn_if_noexec_dir "$CARGO_HOME"; then
    CARGO_HOME="$HOME/.cargo"
    RUSTUP_HOME="$HOME/.rustup"
    export CARGO_HOME RUSTUP_HOME
  fi
  if ! warn_if_noexec_dir "$HOME"; then
    echo "\$HOME ($HOME) is also noexec; cannot install a runnable Rust toolchain." >&2
    echo "Load a cluster Rust module (module load rust) or ask admins for an exec home." >&2
    exit 1
  fi

  echo "Installing Rust toolchain via rustup into:"
  echo "  CARGO_HOME=$CARGO_HOME"
  echo "  RUSTUP_HOME=$RUSTUP_HOME"
  mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
  local rustup_init
  rustup_init="$(mktemp)"
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$rustup_init"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$rustup_init" https://sh.rustup.rs
  else
    echo "Need curl or wget to install rustup." >&2
    exit 1
  fi
  sh "$rustup_init" -y --profile minimal --default-toolchain stable --no-modify-path
  rm -f "$rustup_init"
  source_cargo_env
  if ! rust_works; then
    echo "rustup finished but rustc/cargo are not executable." >&2
    echo "  which rustc: $(command -v rustc 2>/dev/null || echo missing)" >&2
    echo "  which cargo: $(command -v cargo 2>/dev/null || echo missing)" >&2
    echo "This usually means the install path is on a noexec filesystem." >&2
    exit 1
  fi
  echo "Rust installed: $(rustc --version); $(cargo --version)"
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
  ensure_python_packages

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

  ensure_rust
  # Drop a previous in-tree target/ that cannot execute build-scripts (noexec).
  if [[ -d "$TURBOVEC_DIR/target" ]]; then
    echo "Removing noexec in-tree cargo target: $TURBOVEC_DIR/target"
    rm -rf "$TURBOVEC_DIR/target"
  fi
  mkdir -p "$CARGO_TARGET_DIR"
  if ! warn_if_noexec_dir "$CARGO_TARGET_DIR"; then
    echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR is noexec; set CARGO_TARGET_DIR to an exec path (e.g. under \$HOME)." >&2
    exit 1
  fi
  echo "Building turbovec Python wheel (release) ..."
  echo "  CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
  (
    source_cargo_env
    export CARGO_TARGET_DIR
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

run_speed_harness() {
  # Official protocol with mmap: 100k DB + 1k queries, seed=42, k=64, median of 5.
  # TQ is timed first and written out so a later FAISS OOM still leaves TQ numbers.
  local dim="${DATASET#openai-}"
  local tag
  tag="$(dim_tag_for_dataset "$DATASET")"
  local out_json="$TURBOVEC_DIR/benchmarks/results/speed_${tag}_${BITS}bit_${TURBOVEC_ARCH}_${TURBOVEC_THREADS}.json"
  local skip_faiss=0
  [[ -n "$SKIP_FAISS" ]] && skip_faiss=1
  local rayon_threads=1
  local staged=1

  if [[ "$TURBOVEC_THREADS" == st ]]; then
    rayon_threads=1
  else
    rayon_threads="$TURBOVEC_RAYON_THREADS"
  fi
  export RAYON_NUM_THREADS="$rayon_threads"

  # Published ~0.12 ms/query path is staged 4-bit (sign plane → shortlist → exact).
  # TURBOVEC_4BIT_PLANES=0 forces the whole-index scan instead.
  if [[ -n "$TURBOVEC_FULLSCAN" || "$BITS" != "4" ]]; then
    if [[ -n "$TURBOVEC_FULLSCAN" ]]; then
      export TURBOVEC_4BIT_PLANES=0
      staged=0
    else
      # 2-bit has its own staging; leave env alone.
      staged=0
    fi
  else
    unset TURBOVEC_4BIT_PLANES || true
    staged=1
  fi

  echo "Speed harness (mmap, seed=42, n=100000, nq=1000, k=64, median of 5)"
  echo "  arch=$TURBOVEC_ARCH threads=$TURBOVEC_THREADS rayon=$rayon_threads staged=$staged skip_faiss=$skip_faiss"
  if [[ "$BITS" == "4" && "$TURBOVEC_THREADS" == mt && "$DATASET" == openai-1536 && "$staged" == "1" ]]; then
    echo "  published cell: openai-1536 4-bit staged mt (~0.12 ms/query on c3/c4a-standard-8)"
  fi
  echo "  writing $out_json"

  "$PYTHON" - "$dim" "$BITS" "$TURBOVEC_DATA_DIR" "$TURBOVEC_ARCH" "$TURBOVEC_THREADS" \
    "$skip_faiss" "$out_json" "$rayon_threads" "$staged" <<'PY'
import os, sys, time, json
import numpy as np
from turbovec import TurboQuantIndex

dim = int(sys.argv[1])
bits = int(sys.argv[2])
data_dir = sys.argv[3]
arch = sys.argv[4]
threading = sys.argv[5]
skip_faiss = sys.argv[6] == "1"
out_json = sys.argv[7]
rayon_threads = int(sys.argv[8])
staged = sys.argv[9] == "1"

path = os.path.join(data_dir, f"openai-{dim}.npy")
print(f"Loading {path} (mmap) ...", flush=True)
all_vecs = np.load(path, mmap_mode="r")
n_all = int(all_vecs.shape[0])
nbytes = os.path.getsize(path)
print(f"  shape={all_vecs.shape} dtype={all_vecs.dtype} file={nbytes / 1e9:.2f} GB", flush=True)
if n_all < 101_000:
    raise SystemExit(f"need >= 101000 rows, got {n_all}")
if int(all_vecs.shape[1]) != dim:
    raise SystemExit(f"expected dim={dim}, got {all_vecs.shape[1]}")

rng = np.random.RandomState(42)
idx = rng.permutation(n_all)
db_idx = idx[:100_000]
q_idx = idx[100_000:101_000]

# Random fancy-index from a 1M mmap on shared FS is the slow step (minutes).
# Cache the exact seed-42 101k slice under $HOME so re-runs are fast.
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vectorcache-turbovec")
os.makedirs(cache_dir, exist_ok=True)
cache_path = os.path.join(cache_dir, f"openai-{dim}-seed42-101k.npy")
if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
    print(f"Loading cached 101k slice {cache_path} ...", flush=True)
    cached = np.load(cache_path)
    database = cached[:100_000].astype(np.float32, copy=False)
    queries = cached[100_000:101_000].astype(np.float32, copy=False)
    del cached
else:
    print(
        f"Gathering 101k rows via seed=42 permutation from {n_all} "
        f"(random mmap I/O — can take several minutes on shared disks) ...",
        flush=True,
    )
    t_g0 = time.perf_counter()
    # Sort for more sequential reads, then unpermute.
    take = np.concatenate([db_idx, q_idx])
    order = np.argsort(take)
    sorted_take = take[order]
    gathered = np.empty((101_000, dim), dtype=np.float32)
    chunk = 4096
    for start in range(0, 101_000, chunk):
        end = min(start + chunk, 101_000)
        gathered[start:end] = np.asarray(all_vecs[sorted_take[start:end]], dtype=np.float32)
        if start == 0 or end == 101_000 or (end % 16384) == 0:
            print(f"  gathered {end}/101000 rows ({time.perf_counter() - t_g0:.1f}s)", flush=True)
    inv = np.empty(101_000, dtype=np.int64)
    inv[order] = np.arange(101_000)
    selected = gathered[inv]
    database = selected[:100_000]
    queries = selected[100_000:101_000]
    print(f"  gather done in {time.perf_counter() - t_g0:.1f}s; caching to {cache_path}", flush=True)
    np.save(cache_path, selected)
    del gathered, selected
del all_vecs
database /= np.linalg.norm(database, axis=-1, keepdims=True).astype(np.float32)
queries /= np.linalg.norm(queries, axis=-1, keepdims=True).astype(np.float32)
print(f"corpus ready: n={len(database)} nq={len(queries)} dim={dim}", flush=True)

tq = TurboQuantIndex(dim=dim, bit_width=bits)
t_ingest0 = time.perf_counter()
tq.add(database)
tq_ingest_ms = (time.perf_counter() - t_ingest0) * 1000
print(f"tq_ingest_ms={tq_ingest_ms:.3f}", flush=True)
tq.search(queries[:1], k=64)
tq_times = []
for _ in range(5):
    t0 = time.perf_counter()
    tq.search(queries, k=64)
    tq_times.append((time.perf_counter() - t0) / len(queries) * 1000)
tq_ms = sorted(tq_times)[2]
result = {
    "dim": dim,
    "bit_width": bits,
    "arch": arch,
    "threading": threading,
    "rayon_threads": rayon_threads,
    "staged_4bit": staged if bits == 4 else None,
    "n": int(len(database)),
    "nq": int(len(queries)),
    "k": 64,
    "seed": 42,
    "tq_ingest_ms": round(tq_ingest_ms, 3),
    "tq_ms_per_query": round(tq_ms, 3),
    "tq_runs_ms_per_query": [round(t, 3) for t in tq_times],
}
print(json.dumps(result, indent=2), flush=True)
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Wrote TQ result to {out_json}", flush=True)

if skip_faiss:
    print("SKIP_FAISS=1 — done (TQ only).", flush=True)
    raise SystemExit(0)

print("Building FAISS IndexPQFastScan (memory-heavy; may OOM / SIGKILL) ...", flush=True)
import faiss

faiss.omp_set_num_threads(1 if threading == "st" else rayon_threads)

# Match suite/speed_d*_4bit_*: m_pq = dim for 4-bit, m_pq = dim // 2 for 2-bit.
m_pq = dim if bits == 4 else dim // 2
nbits = 4 if bits == 4 else 4  # FastScan scripts use nbits=4 with m matched to rate
if bits == 2:
    # stock 2-bit scripts: IndexPQFastScan(DIM, DIM // 2, 4) — codes pack 2-bit into 4-bit nibbles
    m_pq = dim // 2
    nbits = 4
else:
    m_pq = dim
    nbits = 4

pq = faiss.IndexPQFastScan(dim, m_pq, nbits)
t_ingest0 = time.perf_counter()
pq.train(database)
pq.add(database)
faiss_ingest_ms = (time.perf_counter() - t_ingest0) * 1000
print(f"faiss_ingest_ms={faiss_ingest_ms:.3f}", flush=True)
pq.search(queries[:1], 64)
faiss_times = []
for _ in range(5):
    t0 = time.perf_counter()
    pq.search(queries, 64)
    faiss_times.append((time.perf_counter() - t0) / len(queries) * 1000)
faiss_ms = sorted(faiss_times)[2]
result["faiss_ingest_ms"] = round(faiss_ingest_ms, 3)
result["faiss_ms_per_query"] = round(faiss_ms, 3)
result["faiss_runs_ms_per_query"] = [round(t, 3) for t in faiss_times]
print(json.dumps(result, indent=2), flush=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Wrote full result to {out_json}", flush=True)
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

  echo "=== turbovec bench: DATASET=$DATASET BITS=$BITS SUITE=$TURBOVEC_SUITE ARCH=$TURBOVEC_ARCH THREADS=$TURBOVEC_THREADS RAYON=$TURBOVEC_RAYON_THREADS FULLSCAN=${TURBOVEC_FULLSCAN:-0} ==="

  if [[ "$TURBOVEC_SUITE" == speed && -z "$TURBOVEC_STOCK" ]]; then
    run_speed_harness
    return
  fi

  local script
  script="$(suite_script)"
  if [[ ! -f "$script" ]]; then
    echo "Suite script not found: $script" >&2
    exit 1
  fi
  echo "Running stock script $script"
  "$PYTHON" "$script"
}

main "$@"
