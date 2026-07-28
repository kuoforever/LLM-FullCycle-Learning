[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    python -m fullcycle_bridge `
        --manifest "$projectRoot\fixtures\bridge_v1\valid\runtime-manifest.json" `
        --run-export "$projectRoot\fixtures\bridge_v1\valid\minimal-run-export.json"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
