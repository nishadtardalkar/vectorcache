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

CACHE_ROOT="/rsstu/users/d/dslalush/KneeProject/.cache"
export RUSTUP_HOME="${RUSTUP_HOME:-$CACHE_ROOT/rustup}"
export CARGO_HOME="${CARGO_HOME:-$CACHE_ROOT/cargo}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT}"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME" "$TMPDIR"
export PATH="$CARGO_HOME/bin:$PATH"

if [[ -n "${LD_PRELOAD:-}" ]]; then
    echo "Unsetting LD_PRELOAD for Rust (was: $LD_PRELOAD)"
    unset LD_PRELOAD
fi

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

source_cargo_env() {
    if [[ -f "${CARGO_HOME}/env" ]]; then
        # shellcheck disable=SC1091
        source "${CARGO_HOME}/env"
    fi
    export PATH="$CARGO_HOME/bin:$PATH"
}

rustup_in_cargo_home() {
    [[ -x "$CARGO_HOME/bin/rustup" ]]
}

default_host_triple() {
    case "$(uname -m)" in
        x86_64) echo "x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
        *) echo "$(uname -m)-unknown-linux-gnu" ;;
    esac
}

stable_toolchain_name() {
    echo "stable-$(default_host_triple)"
}

# Some cluster login nodes preload broken libs (NoMachine, etc.) that crash rustc.
rust_cmd() {
    (
        unset LD_PRELOAD
        "$@"
    )
}

stable_rustc() {
    if rustup_in_cargo_home; then
        local rustc
        rustc="$(rust_cmd rustup which rustc 2>/dev/null || true)"
        if [[ -n "$rustc" && -x "$rustc" ]]; then
            echo "$rustc"
            return
        fi
    fi
    echo "$RUSTUP_HOME/toolchains/$(stable_toolchain_name)/bin/rustc"
}

rustc_works() {
    local rustc
    rustc="$(stable_rustc)"
    [[ -x "$rustc" ]] && rust_cmd "$rustc" --version >/dev/null 2>&1
}

cargo_works() {
    rust_cmd cargo --version >/dev/null 2>&1
}

fix_rustup_settings() {
    local host expected
    expected="$(default_host_triple)"
    if [[ ! -f "$RUSTUP_HOME/settings.toml" ]]; then
        return
    fi
    host="$(grep -E '^default_host' "$RUSTUP_HOME/settings.toml" 2>/dev/null | sed 's/.*= *"\([^"]*\)".*/\1/' || true)"
    if [[ -n "$host" && "$host" != "$expected" ]]; then
        echo "Removing stale rustup settings (default_host=$host, expected $expected) ..."
        rm -f "$RUSTUP_HOME/settings.toml"
    fi
}

purge_stable_toolchain() {
    rustup toolchain uninstall stable >/dev/null 2>&1 || true
    rm -rf "$RUSTUP_HOME/toolchains/$(stable_toolchain_name)"
}

install_rustup() {
    echo "Installing rustup into $CARGO_HOME (toolchains in $RUSTUP_HOME) ..."
    fix_rustup_settings
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    source_cargo_env
}

install_stable_toolchain() {
    purge_stable_toolchain
    rust_cmd rustup toolchain install stable --profile minimal
    rust_cmd rustup default stable
}

diagnose_rustc_failure() {
    local rustc
    rustc="$(stable_rustc)"
    echo "=== Rust diagnostics ===" >&2
    echo "Host triple: $(default_host_triple)" >&2
    echo "RUSTUP_HOME=$RUSTUP_HOME" >&2
    echo "CARGO_HOME=$CARGO_HOME" >&2
    echo "LD_PRELOAD=${LD_PRELOAD:-<unset>}" >&2
    if [[ -f "$RUSTUP_HOME/settings.toml" ]]; then
        echo "settings.toml:" >&2
        cat "$RUSTUP_HOME/settings.toml" >&2
    fi
    if [[ -f "$rustc" ]]; then
        ls -la "$rustc" >&2
        echo "rustc --version:" >&2
        rust_cmd "$rustc" --version >&2 2>&1 || true
        if command -v ldd >/dev/null 2>&1; then
            echo "ldd rustc:" >&2
            ldd "$rustc" >&2 2>&1 || true
            echo "glibc: $(ldd --version 2>&1 | head -1)" >&2
        fi
    else
        echo "rustc not found at $rustc" >&2
    fi
    if command -v df >/dev/null 2>&1; then
        echo "disk: $(df -h "$RUSTUP_HOME" 2>/dev/null | tail -1 || true)" >&2
    fi
    echo "========================" >&2
}

ensure_rust() {
    source_cargo_env

    if cargo_works && rustc_works; then
        echo "Rust already installed: $(rust_cmd cargo --version)"
        return
    fi

    if rustup_in_cargo_home; then
        echo "Rustup found in $CARGO_HOME; installing stable toolchain ..."
        if ! install_stable_toolchain || ! rustc_works; then
            echo "Stable toolchain looks corrupt; reinstalling ..."
            install_stable_toolchain
        fi
        source_cargo_env
        if cargo_works && rustc_works; then
            echo "Rust ready: $(rust_cmd cargo --version)"
            return
        fi
    fi

    install_rustup

    if ! rustc_works; then
        echo "Repairing stable toolchain in $RUSTUP_HOME ..."
        install_stable_toolchain
        source_cargo_env
    fi

    if ! cargo_works || ! rustc_works; then
        diagnose_rustc_failure
        cat >&2 <<EOF
Rust installation did not produce a working rustc/cargo.

Common fixes on clusters:
  unset LD_PRELOAD
  rm -rf "$RUSTUP_HOME/toolchains/$(stable_toolchain_name)"
  bash scripts/setup.sh

If rustc fails with a glibc error, load a newer gcc module or ask admins
for a system Rust module instead of rustup.
EOF
        exit 1
    fi

    echo "Rust installed: $(rust_cmd cargo --version)"
}

find_python() {
    local candidates=()

    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        candidates+=("${VIRTUAL_ENV}/bin/python")
    fi
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        candidates+=("$ROOT/.venv/bin/python")
    fi
    if [[ -x "$ROOT/env/bin/python" ]]; then
        candidates+=("$ROOT/env/bin/python")
    fi
    if command -v python3 >/dev/null 2>&1; then
        candidates+=("python3")
    fi
    if command -v python >/dev/null 2>&1; then
        candidates+=("python")
    fi

    local py
    for py in "${candidates[@]}"; do
        if "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            echo "$py"
            return
        fi
    done
}

ensure_python() {
    PYTHON="$(find_python || true)"
    if [[ -z "$PYTHON" ]]; then
        cat >&2 <<'EOF'
Python 3.10+ not found.

Install Python 3.10+ using your system package manager or a module load, then re-run
this script. On clusters, a local venv often works:

  python3 -m venv .venv
  source .venv/bin/activate
  ./scripts/setup.sh
EOF
        exit 1
    fi

    echo "Using Python: $PYTHON ($($PYTHON --version))"
}

ensure_pip() {
    if "$PYTHON" -m pip --version >/dev/null 2>&1; then
        return
    fi

    echo "pip not found for $PYTHON; bootstrapping with ensurepip ..."
    if ! "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
        cat >&2 <<EOF
Could not bootstrap pip for $PYTHON.

Recreate the virtualenv with pip included, for example:
  rm -rf .venv
  python3 -m venv .venv
  source .venv/bin/activate
  ./scripts/setup.sh

Or install pip for your system Python (e.g. python3-pip / pip module).
EOF
        exit 1
    fi
}

install_python_deps() {
    step "Installing Python packages"
    ensure_pip
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
echo "Rust cache: RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME TMPDIR=$TMPDIR"

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
