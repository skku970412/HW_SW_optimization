param(
    [switch]$SkipQor
)

$ErrorActionPreference = "Stop"

Write-Host "[1/9] base validation"
powershell -ExecutionPolicy Bypass -File scripts/run_validation_10x.ps1

Write-Host "[2/9] run tests"
python -m pytest tests -q

if (-not $SkipQor) {
    Write-Host "[3/9] run vivado qor"
    powershell -ExecutionPolicy Bypass -File scripts/run_vivado_qor.ps1 -Part xck26-sfvc784-2LV-c -ClockPeriodNs 5.0
} else {
    Write-Host "[3/9] skip vivado qor"
}

Write-Host "[4/9] accuracy eval"
python scripts/eval_accuracy.py

Write-Host "[5/9] scale-up + onnx integration"
python scripts/run_scaleup_proxy.py
python scripts/run_onnx_integration.py

Write-Host "[6/9] benchmark suite"
python scripts/run_benchmark_suite.py

Write-Host "[7/9] portfolio packaging"
python scripts/generate_portfolio_assets.py

Write-Host "[8/9] portfolio packaging test"
python -m pytest tests/unit/test_portfolio_packaging.py -q

Write-Host "[9/9] refresh boardless status snapshot"
python scripts/summarize_boardless_progress.py

Write-Host "reproduce_portfolio complete"
