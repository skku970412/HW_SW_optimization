param(
    [switch]$SkipQor
)

$ErrorActionPreference = "Stop"

Write-Host "[1/7] base validation"
powershell -ExecutionPolicy Bypass -File scripts/run_validation_10x.ps1

Write-Host "[2/7] run tests"
python -m pytest tests -q

if (-not $SkipQor) {
    Write-Host "[3/7] run vivado qor"
    powershell -ExecutionPolicy Bypass -File scripts/run_vivado_qor.ps1 -Part xck26-sfvc784-2LV-c -ClockPeriodNs 5.0
} else {
    Write-Host "[3/7] skip vivado qor"
}

Write-Host "[4/7] accuracy eval"
python scripts/eval_accuracy.py

Write-Host "[5/7] scale-up + onnx integration"
python scripts/run_scaleup_proxy.py
python scripts/run_onnx_integration.py

Write-Host "[6/7] benchmark suite"
python scripts/run_benchmark_suite.py

Write-Host "[7/8] portfolio packaging"
python scripts/generate_portfolio_assets.py

Write-Host "[8/8] portfolio packaging test"
python -m pytest tests/unit/test_portfolio_packaging.py -q

Write-Host "reproduce_portfolio complete"
