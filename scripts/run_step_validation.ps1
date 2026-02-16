param(
    [Parameter(Mandatory = $true)]
    [string]$StepName,

    [Parameter(Mandatory = $true)]
    [string]$Command,

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
$report = Join-Path $root "results/step_validation_runs.csv"

if (!(Test-Path $report)) {
    "step_name,iteration,timestamp_utc,status,exit_code,pass_count,target_passes,max_runs,command" | Set-Content -Encoding utf8 $report
}

$passCount = 0
$attempt = 0
$failed = $false

while ($attempt -lt $MaxRuns -and $passCount -lt $TargetPasses) {
    $attempt++
    Write-Host ("[{0}] Run {1}/{2} validating... (pass {3}/{4})" -f $StepName, $attempt, $MaxRuns, $passCount, $TargetPasses)
    Invoke-Expression $Command
    $exitCode = $LASTEXITCODE
    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
    if ($exitCode -eq 0) {
        $passCount++
    } else {
        $failed = $true
    }

    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $cmdEscaped = $Command.Replace(",", ";")
    Add-Content -Encoding utf8 $report ("{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f $StepName, $attempt, $timestamp, $status, $exitCode, $passCount, $TargetPasses, $MaxRuns, $cmdEscaped)

    if ($failed) {
        break
    }
}

if (-not $failed -and $passCount -ge $TargetPasses) {
    Write-Host ("[{0}] Validation success: {1} pass(es) achieved in {2} run(s). Max allowed runs: {3}" -f $StepName, $passCount, $attempt, $MaxRuns)
    exit 0
}

Write-Error ("[{0}] Validation failed or target passes not reached. Check results/step_validation_runs.csv." -f $StepName)
exit 1
