# Ingest a sample of vectors from the default dataset (glove) and report variance.
param(
    [int]$Limit = 100,
    [string]$DataDir = "data"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:VECTORCACHE_DATA_DIR = $DataDir

$SampleNpy = Join-Path $DataDir ".cache\glove-sample-$Limit.npy"

Write-Host "Preparing $Limit GloVe vectors in $SampleNpy ..."
python (Join-Path $PSScriptRoot "prepare_glove_sample.py") --data-dir $DataDir --limit $Limit --out $SampleNpy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Ingesting $Limit vectors from glove ..."
cargo run --release --bin ingest-sample -- --npy $SampleNpy --limit $Limit
exit $LASTEXITCODE
