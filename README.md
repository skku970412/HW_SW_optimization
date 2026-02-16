# Transformer Acceleration (Boardless LLM Inference)

Repository for boardless development, validation, and portfolio packaging
of an FPGA/NPU-style LLM inference accelerator.

## Current Status

- Boardless track: B1~B8 PASS
- Optimization round: N1~N6 PASS
- tiny_cpu_tps: 4997.085077
- fpga_est_tps: 171.326754
- scaleup_proxy_tps: 99.776003
- speedup_fpga_est_vs_scaleup_proxy: 1.717114 (primary)
- onnx_mae_avg: 0.047009

## Quick Start

```powershell
python -m pip install -r requirements-boardless.txt
powershell -ExecutionPolicy Bypass -File scripts/run_validation_10x.ps1
python -m pytest tests -q
```

## Reproduce Portfolio Package

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1
```

## P1 RTL Backend Path

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_n7_rtl_backend.ps1 -MaxRuns 10
python scripts/run_rtl_backend_flow.py
```

## Outputs

- Final report: `docs/portfolio/final_report.md`
- Figures: `docs/portfolio/figures/`
- Manifest: `docs/portfolio/manifest.json`
- Runbook: `docs/portfolio/runbook.md`

## Notes

- `results/` and `logs/` are ignored by default and regenerated per run.
- Commit-facing portfolio artifacts are under `docs/portfolio/`.

## Implementation Scope (Current)

- This repository currently uses proxy RTL kernels for boardless pre-silicon bring-up.
- Runtime default path is NumPy backend; RTL path is validated via cocotb/unit tests.
- Main KPI for cross-scale fairness is `fpga_est_tps / scaleup_proxy_tps`.

## Visualization Results

### Throughput
![Throughput](docs/portfolio/figures/performance_tps.png)

### QoR Resources
![QoR](docs/portfolio/figures/qor_resources.png)

### ONNX MAE
![ONNX MAE](docs/portfolio/figures/onnx_mae.png)
