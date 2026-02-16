param(
    [string]$Name = "latest",
    [switch]$IncludeCode = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$deliverables = Join-Path $root "deliverables"
$pkgRoot = Join-Path $deliverables ("review_package_" + $Name)
$zipPath = Join-Path $deliverables ("transformer_acc_review_" + $Name + ".zip")

if (Test-Path $pkgRoot) {
    Remove-Item $pkgRoot -Recurse -Force
}
New-Item -ItemType Directory -Force $pkgRoot | Out-Null

$files = @(
    "README.md",
    ".gitignore",
    "pytest.ini",
    "requirements-boardless.txt",
    "docs/boardless_track.md",
    "docs/milestones.md",
    "docs/spec.md",
    "docs/stack_decision.md",
    "docs/portfolio/final_report.md",
    "docs/portfolio/manifest.json",
    "docs/portfolio/runbook.md",
    "docs/portfolio/figures/performance_tps.png",
    "docs/portfolio/figures/qor_resources.png",
    "docs/portfolio/figures/onnx_mae.png",
    "scripts/reproduce_portfolio.ps1",
    "scripts/run_n6_packaging.ps1",
    "scripts/run_n7_rtl_backend.ps1",
    "scripts/generate_portfolio_assets.py",
    "scripts/validate_project.py",
    "results/benchmark_suite.csv",
    "results/benchmark_template.csv",
    "results/benchmark_template.md",
    "results/qor_summary.csv",
    "results/boardless_status.md",
    "results/rtl_backend_flow_result.json",
    "results/step_validation_runs.csv",
    "results/boardless_progress_log.csv"
)

foreach ($f in $files) {
    $src = Join-Path $root $f
    if (Test-Path $src) {
        $dst = Join-Path $pkgRoot $f
        $dstDir = Split-Path -Parent $dst
        New-Item -ItemType Directory -Force $dstDir | Out-Null
        Copy-Item -Path $src -Destination $dst -Force
    }
}

function Should-SkipFile {
    param([string]$PathText)

    $p = $PathText.Replace("/", "\")
    $skipPatterns = @(
        "*\__pycache__\*",
        "*\.pytest_cache\*",
        "*\sim_build\*",
        "*\logs\*",
        "*\results\*",
        "*\deliverables\*",
        "*\.Xil\*",
        "*\tests\vectors\*.npy",
        "*\tests\tb\*.out",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.vcd",
        "*.fst",
        "*.lxt",
        "*.vvp",
        "*.vpi",
        "*.wdb",
        "*.wcfg",
        "*.jou",
        "*.log",
        "*.zip"
    )

    foreach ($pat in $skipPatterns) {
        if ($p -like $pat) {
            return $true
        }
    }
    return $false
}

if ($IncludeCode) {
    $codeDirs = @("hw", "sw", "runtime", "scripts", "tests", "docs")
    foreach ($dir in $codeDirs) {
        $srcDir = Join-Path $root $dir
        if (!(Test-Path $srcDir)) {
            continue
        }

        Get-ChildItem -Path $srcDir -Recurse -File | ForEach-Object {
            $srcPath = $_.FullName
            if (Should-SkipFile -PathText $srcPath) {
                return
            }

            $relPath = $srcPath.Substring($root.Length + 1)
            $dstPath = Join-Path $pkgRoot $relPath
            $dstDir = Split-Path -Parent $dstPath
            New-Item -ItemType Directory -Force $dstDir | Out-Null
            Copy-Item -Path $srcPath -Destination $dstPath -Force
        }
    }
}

$guide = @(
    "# Review Guide",
    "",
    "## Open Order",
    "1. README.md",
    "2. docs/portfolio/final_report.md",
    "3. docs/portfolio/figures/*.png",
    "4. results/benchmark_suite.csv",
    "5. results/qor_summary.csv",
    "6. results/boardless_status.md",
    "7. hw/, sw/, runtime/, scripts/, tests/ source code",
    "",
    "## Reproduction",
    "- powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1",
    "- powershell -ExecutionPolicy Bypass -File scripts/run_n6_packaging.ps1 -MaxRuns 10",
    "",
    "## Notes",
    "- Curated package for decision-maker review.",
    "- Includes portfolio artifacts, KPI summaries, validation logs, and source code.",
    "- Excludes caches/logs/generated binaries by default."
)
Set-Content -Path (Join-Path $pkgRoot "REVIEW_GUIDE.md") -Value $guide -Encoding utf8

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $pkgRoot "*") -DestinationPath $zipPath -Force

$zipItem = Get-Item $zipPath
Write-Host ("ZIP_PATH=" + $zipItem.FullName)
Write-Host ("ZIP_SIZE_BYTES=" + $zipItem.Length)
exit 0
