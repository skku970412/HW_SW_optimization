param(
    [ValidateRange(1, 10)]
    [int]$MaxRuns = 10
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$stepName = "N9_cycle_model_calibration"
$command = "python -m pytest tests/unit/test_cycle_model_calibration.py -q"

Write-Host "[N9] run cycle-model calibration validation (max $MaxRuns runs, stop on first PASS)"
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
        --week N9 `
        --step $stepName `
        --status PASS `
        --validation-runs $runs `
        --summary "Cycle model calibration (RTL observed counters vs runtime model) passed." | Out-Null
    exit 0
}

python (Join-Path $root "scripts/log_boardless_progress.py") `
    --week N9 `
    --step $stepName `
    --status FAIL `
    --validation-runs $runs `
    --summary "N9 cycle model calibration validation failed." `
    --blocker "See results/step_validation_runs.csv for $stepName." | Out-Null
exit 1
