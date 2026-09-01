# Install toolchain and dependencies for vectorcache.
#
# Usage:
#   .\scripts\setup.ps1                  # Rust + Python (default workflow)
#   .\scripts\setup.ps1 -WithGlove         # also install HDF5 for the Rust glove feature
#   .\scripts\setup.ps1 -FetchDatasets     # download benchmark datasets after setup
#   .\scripts\setup.ps1 -SkipBuild         # install deps only, skip cargo build
#
param(
    [switch]$WithGlove,
    [switch]$FetchDatasets,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Command([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Rust {
    if (Test-Command "cargo") {
        $version = (cargo --version)
        Write-Host "Rust already installed: $version"
        return
    }

    Write-Host "Rust not found. Installing via rustup ..."
    $rustup = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustup
    & $rustup -y --default-toolchain stable
    if ($LASTEXITCODE -ne 0) { throw "rustup-init failed with exit code $LASTEXITCODE" }

    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if ($env:PATH -notlike "*$cargoBin*") {
        $env:PATH = "$cargoBin;$env:PATH"
        Write-Host "Added $cargoBin to PATH for this session."
        Write-Host "Restart your terminal (or log out/in) so cargo is on PATH permanently."
    }
}

function Get-PythonCommand {
    if (Test-Command "python") {
        return "python"
    }
    if (Test-Command "py") {
        return "py -3"
    }
    return $null
}

function Ensure-Python {
    $python = Get-PythonCommand
    if (-not $python) {
        throw @"
Python 3 not found.

Install Python 3.10+ from https://www.python.org/downloads/
Enable 'Add python.exe to PATH' during installation, then re-run this script.
"@
    }

    $version = Invoke-Expression "$python --version"
    Write-Host "Python already installed: $version"
    return $python
}

function Install-PythonDeps([string]$Python) {
    Write-Step "Installing Python packages"
    $requirements = Join-Path $PSScriptRoot "requirements.txt"
    Invoke-Expression "$Python -m pip install --upgrade pip"
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    Invoke-Expression "$Python -m pip install -r `"$requirements`""
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

function Find-Hdf5Root {
    if ($env:HDF5_DIR -and (Test-Path $env:HDF5_DIR)) {
        return $env:HDF5_DIR
    }

    $candidates = @(
        "$env:VCPKG_ROOT\installed\x64-windows",
        "$env:LOCALAPPDATA\vcpkg\installed\x64-windows",
        "C:\vcpkg\installed\x64-windows",
        "C:\Program Files\HDF_Group\HDF5\1.14.5",
        "C:\Program Files\HDF_Group\HDF5\1.14.4",
        "C:\Program Files\HDF_Group\HDF5\1.12.2"
    )

    foreach ($path in $candidates) {
        if ($path -and (Test-Path (Join-Path $path "include\hdf5.h"))) {
            return $path
        }
    }

    return $null
}

function Ensure-Hdf5 {
    $root = Find-Hdf5Root
    if ($root) {
        $env:HDF5_DIR = $root
        Write-Host "Using HDF5 at $root"
        return
    }

    $vcpkgRoot = $env:VCPKG_ROOT
    if (-not $vcpkgRoot) {
        $vcpkgRoot = Join-Path $env:LOCALAPPDATA "vcpkg"
    }

    if (-not (Test-Path (Join-Path $vcpkgRoot "vcpkg.exe"))) {
        Write-Step "Installing HDF5 via vcpkg (one-time; may take several minutes)"
        if (-not (Test-Path $vcpkgRoot)) {
            git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
        }
        & (Join-Path $vcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
        if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed" }
    }

    & (Join-Path $vcpkgRoot "vcpkg.exe") install hdf5:x64-windows
    if ($LASTEXITCODE -ne 0) { throw "vcpkg install hdf5 failed" }

    $env:HDF5_DIR = Join-Path $vcpkgRoot "installed\x64-windows"
    if (-not (Test-Path (Join-Path $env:HDF5_DIR "include\hdf5.h"))) {
        throw "HDF5 install finished but headers were not found under $($env:HDF5_DIR)"
    }

    Write-Host "HDF5 installed. Set HDF5_DIR=$($env:HDF5_DIR) for future builds."
    Write-Host "Add to your profile:  `$env:HDF5_DIR = '$($env:HDF5_DIR)'"
}

function Build-Project {
    Write-Step "Fetching Rust crates"
    cargo fetch
    if ($LASTEXITCODE -ne 0) { throw "cargo fetch failed" }

    if ($WithGlove) {
        Write-Step "Building with glove feature (requires HDF5)"
        cargo build --features glove
    } else {
        Write-Step "Building default features"
        cargo build
    }

    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

    Write-Step "Running tests"
    if ($WithGlove) {
        cargo test --features glove
    } else {
        cargo test
    }
    if ($LASTEXITCODE -ne 0) { throw "cargo test failed" }
}

function Ensure-DataDir {
    $dataDir = Join-Path $Root "data"
    if (-not (Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir | Out-Null
        Write-Host "Created $dataDir"
    }
}

function Fetch-Datasets {
    Write-Step "Downloading benchmark datasets"
    cargo run --release --bin fetch-datasets -- all
    if ($LASTEXITCODE -ne 0) { throw "fetch-datasets failed" }
}

Write-Step "vectorcache setup"
Write-Host "Project root: $Root"

Ensure-Rust
$python = Ensure-Python
Install-PythonDeps $python
Ensure-DataDir

if ($WithGlove) {
    Ensure-Hdf5
} else {
    Write-Host ""
    Write-Host "Skipping HDF5 (not needed for the default Python-based GloVe workflow)."
    Write-Host "Re-run with -WithGlove to build the native Rust GloVe HDF5 reader."
}

if (-not $SkipBuild) {
    Build-Project
}

if ($FetchDatasets) {
    Fetch-Datasets
}

Write-Step "Setup complete"
Write-Host @"

Next steps:
  cargo run --release --bin ingest-sample -- --npy data/.cache/glove-sample-100.npy
  .\scripts\ingest-glove-sample.ps1 -Limit 100
  cargo run --release --bin fetch-datasets -- glove

Optional env defaults live in scripts/vectorcache.env
"@
