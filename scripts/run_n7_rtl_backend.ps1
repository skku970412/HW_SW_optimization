param(
    [ValidateRange(1, 10)]
    [int]$MaxRuns = 10
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$stepName = "N7_rtl_backend_p1"
$command = "python -m pytest tests/unit/test_rtl_backend.py tests/unit/test_rtl_backend_flow.py tests/tb/test_rtl_compile.py tests/tb/test_cocotb_runner.py -q"

Write-Host "[N7] run RTL-backend P1 validation (max $MaxRuns runs, stop on first PASS)"
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
        --week N7 `
        --step $stepName `
        --status PASS `
        --validation-runs $runs `
        --summary "P1 runtime RTL backend + npu_top + compile/cocotb/unit validation passed." | Out-Null
    exit 0
}

python (Join-Path $root "scripts/log_boardless_progress.py") `
    --week N7 `
    --step $stepName `
    --status FAIL `
    --validation-runs $runs `
    --summary "N7 RTL backend validation failed." `
    --blocker "See results/step_validation_runs.csv for $stepName." | Out-Null
exit 1
