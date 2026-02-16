param(
    [string]$Part = "xck26-sfvc784-2LV-c",
    [double]$ClockPeriodNs = 5.0
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $root "results/qor"
$tcl = Join-Path $root "scripts/vivado/run_qor_single.tcl"
$parser = Join-Path $root "scripts/parse_vivado_qor.py"
$summaryCsv = Join-Path $root "results/qor_summary.csv"

function Find-VivadoCommand {
    $cmd = Get-Command vivado -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $roots = @(
        "D:\Xilinx\Vivado",
        "C:\Xilinx\Vivado",
        "D:\AMD\Vivado",
        "C:\AMD\Vivado"
    )
    foreach ($r in $roots) {
        if (!(Test-Path $r)) { continue }
        $versions = Get-ChildItem $r -Directory | Sort-Object Name -Descending
        foreach ($v in $versions) {
            $bin = Join-Path $v.FullName "bin"
            $cand = @(
                (Join-Path $bin "vivado.bat"),
                (Join-Path $bin "vivado.exe"),
                (Join-Path $bin "vivado")
            )
            foreach ($c in $cand) {
                if (Test-Path $c) { return $c }
            }
        }
    }
    return $null
}

function Invoke-VivadoBatch {
    param(
        [string]$VivadoCmd,
        [string]$Top
    )
    $topOut = Join-Path $outDir $Top
    New-Item -ItemType Directory -Force $topOut | Out-Null
    $logFile = Join-Path $topOut "vivado_run.log"

    if ($VivadoCmd.ToLower().EndsWith(".bat")) {
        cmd /c "`"$VivadoCmd`" -mode batch -source `"$tcl`" -notrace -tclargs `"$root`" `"$outDir`" `"$Part`" `"$Top`" `"$ClockPeriodNs`"" > $logFile 2>&1
    } else {
        & $VivadoCmd -mode batch -source $tcl -notrace -tclargs $root $outDir $Part $Top $ClockPeriodNs *> $logFile
    }

    $ok = ($LASTEXITCODE -eq 0)
    if (-not $ok) {
        $tail = Get-Content -Path $logFile -Tail 40
        Write-Host "Vivado run failed for top=$Top"
        $tail | ForEach-Object { Write-Host $_ }
    }
    return $ok
}

$vivadoCmd = Find-VivadoCommand
if (-not $vivadoCmd) {
    Write-Error "vivado command not found."
    exit 2
}

$vivadoBin = Split-Path -Parent $vivadoCmd
$env:PATH = "$vivadoBin;$env:PATH"

$tops = @("gemm_core", "attention_core", "kv_cache", "decoder_block_top", "npu_top")
New-Item -ItemType Directory -Force $outDir | Out-Null

foreach ($top in $tops) {
    Write-Host "Running Vivado QoR for top=$top part=$Part"
    $ok = Invoke-VivadoBatch -VivadoCmd $vivadoCmd -Top $top
    if (-not $ok) {
        exit 3
    }
}

python $parser --qor-dir $outDir --out $summaryCsv --part $Part --clock-period-ns $ClockPeriodNs
if ($LASTEXITCODE -ne 0) {
    Write-Error "QoR parsing failed."
    exit 4
}

Write-Host "QoR completed. Summary: $summaryCsv"
exit 0
