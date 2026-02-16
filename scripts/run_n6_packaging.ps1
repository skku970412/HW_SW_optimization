param(
    [ValidateRange(1, 10)]
    [int]$MaxRuns = 10
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$stepName = "N6_portfolio_packaging"
$command = "python -m pytest tests/unit/test_portfolio_packaging.py -q"

Write-Host "[N6] generate portfolio assets"
python (Join-Path $root "scripts/generate_portfolio_assets.py")
if ($LASTEXITCODE -ne 0) {
    python (Join-Path $root "scripts/log_boardless_progress.py") `
        --week N6 `
        --step $stepName `
        --status FAIL `
        --validation-runs 1 `
        --summary "Portfolio asset generation failed." `
        --blocker "scripts/generate_portfolio_assets.py returned non-zero." | Out-Null
    exit 1
}

Write-Host "[N6] validate packaging (max $MaxRuns runs, stop on first PASS)"
powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts/run_step_validation.ps1") `
    -StepName $stepName `
    -Command $command `
    -MaxRuns $MaxRuns `
    -TargetPasses 1

$csv = Join-Path $root "results/step_validation_runs.csv"
$rows = Import-Csv $csv | Where-Object { $_.step_name -eq $stepName }
$last = $rows[-1]
$status = $last.status
$runs = [int]$last.iteration

if ($status -eq "PASS") {
    python (Join-Path $root "scripts/log_boardless_progress.py") `
        --week N6 `
        --step $stepName `
        --status PASS `
        --validation-runs $runs `
        --summary "Automated README/figures/final report/manifest/runbook packaging passed." | Out-Null
    exit 0
}

python (Join-Path $root "scripts/log_boardless_progress.py") `
    --week N6 `
    --step $stepName `
    --status FAIL `
    --validation-runs $runs `
    --summary "N6 packaging validation failed." `
    --blocker "See results/step_validation_runs.csv for $stepName." | Out-Null
exit 1
