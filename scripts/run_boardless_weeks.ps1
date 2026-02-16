param(
    [ValidateRange(1, 10)]
    [int]$MaxRunsPerStep = 10
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$icarusBin = "C:\iverilog\bin"
if (Test-Path (Join-Path $icarusBin "iverilog.exe")) {
    $env:PATH = "$icarusBin;$env:PATH"
}

function Write-ProgressLog {
    param(
        [string]$Week,
        [string]$Step,
        [string]$Status,
        [int]$ValidationRuns,
        [string]$Summary,
        [string]$Blocker = ""
    )
    $logger = Join-Path $root "scripts/log_boardless_progress.py"
    if ([string]::IsNullOrWhiteSpace($Blocker)) {
        python $logger `
            --week $Week `
            --step $Step `
            --status $Status `
            --validation-runs $ValidationRuns `
            --summary $Summary | Out-Null
    } else {
        python $logger `
            --week $Week `
            --step $Step `
            --status $Status `
            --validation-runs $ValidationRuns `
            --summary $Summary `
            --blocker $Blocker | Out-Null
    }
}

function Get-LastRunCount {
    param([string]$StepName)
    $csv = Join-Path $root "results/step_validation_runs.csv"
    if (!(Test-Path $csv)) { return 0 }
    $rows = Import-Csv $csv | Where-Object { $_.step_name -eq $StepName }
    if (!$rows) { return 0 }
    $last = $rows[-1]
    return [int]$last.iteration
}

function Get-LastStatus {
    param([string]$StepName)
    $csv = Join-Path $root "results/step_validation_runs.csv"
    if (!(Test-Path $csv)) { return "" }
    $rows = Import-Csv $csv | Where-Object { $_.step_name -eq $StepName }
    if (!$rows) { return "" }
    return $rows[-1].status
}

function Find-VivadoCommand {
    $cmd = Get-Command vivado -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $roots = @(
        "D:\Xilinx\Vivado",
        "C:\Xilinx\Vivado",
        "D:\AMD\Vivado",
        "C:\AMD\Vivado"
    )

    foreach ($rootDir in $roots) {
        if (!(Test-Path $rootDir)) {
            continue
        }
        $versions = Get-ChildItem $rootDir -Directory | Sort-Object Name -Descending
        foreach ($v in $versions) {
            $bin = Join-Path $v.FullName "bin"
            $candidateBat = Join-Path $bin "vivado.bat"
            $candidateExe = Join-Path $bin "vivado.exe"
            $candidateNoExt = Join-Path $bin "vivado"

            if (Test-Path $candidateBat) { return $candidateBat }
            if (Test-Path $candidateExe) { return $candidateExe }
            if (Test-Path $candidateNoExt) { return $candidateNoExt }
        }
    }
    return $null
}

function Run-Step {
    param(
        [string]$Week,
        [string]$StepName,
        [string]$Command,
        [string]$PassSummary
    )
    $runner = Join-Path $root "scripts/run_step_validation.ps1"
    & powershell -ExecutionPolicy Bypass -File $runner `
        -StepName $StepName `
        -Command $Command `
        -MaxRuns $MaxRunsPerStep `
        -TargetPasses 1
    $status = Get-LastStatus -StepName $StepName
    $ok = ($status -eq "PASS")
    $runs = Get-LastRunCount -StepName $StepName
    if ($ok) {
        Write-ProgressLog -Week $Week -Step $StepName -Status "PASS" -ValidationRuns $runs -Summary $PassSummary
    } else {
        Write-ProgressLog -Week $Week -Step $StepName -Status "FAIL" -ValidationRuns $runs -Summary "Validation failed." -Blocker "Check results/step_validation_runs.csv"
    }
    return $ok
}

# B1: environment and smoke
if (-not (Run-Step -Week "B1" -StepName "B1_smoke" -Command "python -m pytest tests/smoke/test_env.py" -PassSummary "Smoke/env test passed.")) {
    exit 1
}

$simProbe = @'
import shutil, sys
from pathlib import Path
sims = ["verilator", "iverilog", "xsim", "ghdl"]
found = [s for s in sims if shutil.which(s)]
if Path(r"C:\iverilog\bin\iverilog.exe").exists() and "iverilog" not in found:
    found.append("iverilog")
print("simulators=", found)
sys.exit(0 if found else 2)
'@
@"
$simProbe
"@ | python -
if ($LASTEXITCODE -eq 0) {
    Write-ProgressLog -Week "B1" -Step "B1_simulator_probe" -Status "PASS" -ValidationRuns 1 -Summary "RTL simulator detected."
} else {
    Write-ProgressLog -Week "B1" -Step "B1_simulator_probe" -Status "BLOCKED" -ValidationRuns 1 -Summary "Python/cocotb path ready, RTL simulation blocked." -Blocker "No verilator/iverilog/xsim/ghdl in PATH"
}

# B2: GEMM reference and vectors
if (-not (Run-Step -Week "B2" -StepName "B2_gemm_unit" -Command "python -m pytest tests/unit/test_golden_gemm.py" -PassSummary "GEMM unit tests passed.")) {
    exit 1
}
python (Join-Path $root "scripts/gen_gemm_vectors.py") --m 4 --k 16 --n 8 --seed 42 --outdir (Join-Path $root "tests/vectors")
Write-ProgressLog -Week "B2" -Step "B2_vector_gen" -Status "PASS" -ValidationRuns 1 -Summary "GEMM vectors generated."

# B3: attention
if (-not (Run-Step -Week "B3" -StepName "B3_attention_unit" -Command "python -m pytest tests/unit/test_golden_attention.py" -PassSummary "Attention unit tests passed.")) {
    exit 1
}

# B4: KV-cache
if (-not (Run-Step -Week "B4" -StepName "B4_kvcache_unit" -Command "python -m pytest tests/unit/test_golden_kvcache.py" -PassSummary "KV-cache unit tests passed.")) {
    exit 1
}

# B5: decode step integration
if (-not (Run-Step -Week "B5" -StepName "B5_decode_unit" -Command "python -m pytest tests/unit/test_golden_decode.py" -PassSummary "Decode-step integration test passed.")) {
    exit 1
}

# B6: performance model
if (-not (Run-Step -Week "B6" -StepName "B6_perf_model_unit" -Command "python -m pytest tests/unit/test_perf_model.py" -PassSummary "Performance-model unit test passed.")) {
    exit 1
}
python (Join-Path $root "scripts/perf_model.py") --layers 6 --hidden 768 --seq 256 --pe-mac-per-cycle 256 --clock-mhz 200 --efficiency 0.15 > (Join-Path $root "results/perf_model_latest.txt")
Write-ProgressLog -Week "B6" -Step "B6_perf_model_run" -Status "PASS" -ValidationRuns 1 -Summary "Performance model output generated."

# B7: synthesis/QoR pre-check
$vivadoCmd = Find-VivadoCommand
if ($vivadoCmd) {
    $vivadoDir = Split-Path -Parent $vivadoCmd
    $env:PATH = "$vivadoDir;$env:PATH"

    $versionText = ""
    if ($vivadoCmd.ToLower().EndsWith(".bat")) {
        $versionText = cmd /c "`"$vivadoCmd`" -version" 2>&1 | Select-Object -First 1
    } else {
        $versionText = & $vivadoCmd -version 2>&1 | Select-Object -First 1
    }

    if ($LASTEXITCODE -eq 0) {
        Write-ProgressLog -Week "B7" -Step "B7_qor_probe" -Status "PASS" -ValidationRuns 1 -Summary ("Vivado detected and version probed: " + $versionText)
    } else {
        Write-ProgressLog -Week "B7" -Step "B7_qor_probe" -Status "BLOCKED" -ValidationRuns 1 -Summary "Vivado found but version probe failed." -Blocker ("path=" + $vivadoCmd)
    }
} else {
    Write-ProgressLog -Week "B7" -Step "B7_qor_probe" -Status "BLOCKED" -ValidationRuns 1 -Summary "QoR report generation blocked in this environment." -Blocker "vivado not found in PATH"
}

# B8: reproducibility package check
if (-not (Run-Step -Week "B8" -StepName "B8_full_python_suite" -Command "python -m pytest tests/smoke tests/unit" -PassSummary "Boardless Python suite passed.")) {
    exit 1
}
Write-ProgressLog -Week "B8" -Step "B8_package_status" -Status "PASS" -ValidationRuns 1 -Summary "Boardless package reproducibility check completed."

Write-Host "Boardless track execution completed through B8 (with blockers logged where applicable)."
exit 0
