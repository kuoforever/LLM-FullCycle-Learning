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
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $actualRecords = @(
        python -m fullcycle_bridge.dataset_cli `
            --manifest "$projectRoot\fixtures\bridge_v1\valid\runtime-manifest.json" `
            --run-export "$projectRoot\fixtures\reliability_dataset_v1\inputs\failure-denial-recovery-budget-sequence.json" `
            --run-export "$projectRoot\fixtures\reliability_dataset_v1\inputs\unknown-outcome.json"
    )
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $expectedRecords = @(
        Get-Content -LiteralPath "$projectRoot\fixtures\reliability_dataset_v1\expected-records.jsonl"
    )
    if (Compare-Object -ReferenceObject $expectedRecords -DifferenceObject $actualRecords -SyncWindow 0) {
        throw "Reliability dataset output does not match the pinned JSONL fixture."
    }
    Write-Output "{`"reliability_dataset_fixture_match`":true,`"records`":$($actualRecords.Count)}"
    exit 0
}
finally {
    Pop-Location
}
