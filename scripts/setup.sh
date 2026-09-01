#!/usr/bin/env bash
# Install toolchain and dependencies for vectorcache.
#
# Usage:
#   ./scripts/setup.sh                  # Rust + Python (default workflow)
#   ./scripts/setup.sh --with-glove     # also install HDF5 for the Rust glove feature
#   ./scripts/setup.sh --fetch-datasets # download benchmark datasets after setup
#   ./scripts/setup.sh --skip-build     # install deps only, skip cargo build
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_GLOVE=0
FETCH_DATASETS=0
SKIP_BUILD=0

for arg in "$@"; do
    case "$arg" in
        --with-glove) WITH_GLOVE=1 ;;
        --fetch-datasets) FETCH_DATASETS=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

step() {
    echo
    echo "==> $1"
}

ensure_rust() {
    if command -v cargo >/dev/null 2>&1; then
        echo "Rust already installed: $(cargo --version)"
        return
    fi

    echo "Rust not found. Installing via rustup ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
}

ensure_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        cat >&2 <<'EOF'
Python 3 not found.

Install Python 3.10+ using your system package manager, then re-run this script.
EOF
        exit 1
    fi

    echo "Python already installed: $($PYTHON --version)"
}

install_python_deps() {
    step "Installing Python packages"
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r "$ROOT/scripts/requirements.txt"
}

find_hdf5_root() {
    if [[ -n "${HDF5_DIR:-}" && -f "${HDF5_DIR}/include/hdf5.h" ]]; then
        echo "$HDF5_DIR"
        return
    fi

    local candidates=(
        "/usr/local/hdf5"
        "/opt/homebrew/opt/hdf5"
        "/usr/local/opt/hdf5"
        "/usr"
    )

    for path in "${candidates[@]}"; do
        if [[ -f "${path}/include/hdf5.h" ]]; then
            echo "$path"
            return
        fi
    done

    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists hdf5; then
        pkg-config --variable=prefix hdf5
        return
    fi
}

ensure_hdf5() {
    local root
    root="$(find_hdf5_root || true)"
    if [[ -n "$root" ]]; then
        export HDF5_DIR="$root"
        echo "Using HDF5 at $HDF5_DIR"
        return
    fi

    step "Installing HDF5"
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        brew install hdf5
        export HDF5_DIR="$(brew --prefix hdf5)"
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y libhdf5-dev pkg-config
        export HDF5_DIR="/usr"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y hdf5-devel pkg-config
        export HDF5_DIR="/usr"
    else
        cat >&2 <<'EOF'
Could not install HDF5 automatically.

Install the HDF5 development package for your OS, set HDF5_DIR to the install
prefix (directory containing include/hdf5.h), then re-run with --with-glove.
EOF
        exit 1
    fi

    echo "HDF5 installed. Export HDF5_DIR=$HDF5_DIR for future builds."
}

build_project() {
    step "Fetching Rust crates"
    cargo fetch

    if [[ "$WITH_GLOVE" -eq 1 ]]; then
        step "Building with glove feature (requires HDF5)"
        cargo build --features glove
        cargo test --features glove
    else
        step "Building default features"
        cargo build
        cargo test
    fi
}

ensure_data_dir() {
    mkdir -p "$ROOT/data"
}

fetch_datasets() {
    step "Downloading benchmark datasets"
    cargo run --release --bin fetch-datasets -- all
}

step "vectorcache setup"
echo "Project root: $ROOT"

ensure_rust
ensure_python
install_python_deps
ensure_data_dir

if [[ "$WITH_GLOVE" -eq 1 ]]; then
    ensure_hdf5
else
    echo
    echo "Skipping HDF5 (not needed for the default Python-based GloVe workflow)."
    echo "Re-run with --with-glove to build the native Rust GloVe HDF5 reader."
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    build_project
fi

if [[ "$FETCH_DATASETS" -eq 1 ]]; then
    fetch_datasets
fi

step "Setup complete"
cat <<'EOF'

Next steps:
  cargo run --release --bin ingest-sample -- --npy data/.cache/glove-sample-100.npy
  ./scripts/ingest-glove-sample.ps1   # Windows
  cargo run --release --bin fetch-datasets -- glove

Optional env defaults live in scripts/vectorcache.env
EOF
