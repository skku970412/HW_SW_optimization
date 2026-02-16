param(
    [ValidateRange(1, 10)]
    [int]$MaxRuns = 10,

    [ValidateRange(1, 10)]
    [int]$TargetPasses = 1
)

$ErrorActionPreference = "Stop"

if ($TargetPasses -gt $MaxRuns) {
    Write-Error "TargetPasses는 MaxRuns보다 클 수 없습니다."
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot
$validator = Join-Path $root "scripts/validate_project.py"
$report = Join-Path $root "results/validation_runs.csv"

$python = Get-Command python -ErrorAction SilentlyContinue
$py = Get-Command py -ErrorAction SilentlyContinue

if ($python) {
    $runner = @("python")
} elseif ($py) {
    $runner = @("py", "-3")
} else {
    Write-Error "python 또는 py 실행 파일을 찾을 수 없습니다."
    exit 1
}

"iteration,timestamp_utc,status,exit_code,pass_count,target_passes,max_runs" | Set-Content -Encoding utf8 $report

$passCount = 0
$attempt = 0
$failed = $false

while ($attempt -lt $MaxRuns -and $passCount -lt $TargetPasses) {
    $attempt++
    Write-Host ("[Run {0}/{1}] validating... (pass {2}/{3})" -f $attempt, $MaxRuns, $passCount, $TargetPasses)

    if ($runner.Count -eq 1) {
        & $runner[0] $validator
    } else {
        & $runner[0] $runner[1] $validator
    }

    $exitCode = $LASTEXITCODE
    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
    if ($exitCode -eq 0) {
        $passCount++
    } else {
        $failed = $true
    }

    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Encoding utf8 $report ("{0},{1},{2},{3},{4},{5},{6}" -f $attempt, $timestamp, $status, $exitCode, $passCount, $TargetPasses, $MaxRuns)

    if ($failed) {
        break
    }
}

if (-not $failed -and $passCount -ge $TargetPasses) {
    Write-Host ("Validation success: {0} pass(es) achieved in {1} run(s). Max allowed runs: {2}" -f $passCount, $attempt, $MaxRuns)
    exit 0
}

Write-Error "Validation failed or target passes not reached. Check results/validation_runs.csv and validator output."
exit 1
