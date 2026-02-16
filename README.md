# Transformer Acceleration (Boardless LLM Inference)

Repository for boardless development, validation, and portfolio packaging
of an FPGA/NPU-style LLM inference accelerator.

## Current Status

- Boardless track: B1~B8 PASS
- Optimization round: N1~N6 PASS
- tiny_cpu_tps: 7212.405362
- fpga_est_tps: 171.326754
- scaleup_proxy_tps: 108.536976
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

## Outputs

- Final report: `docs/portfolio/final_report.md`
- Figures: `docs/portfolio/figures/`
- Manifest: `docs/portfolio/manifest.json`
- Runbook: `docs/portfolio/runbook.md`

## Notes

- `results/` and `logs/` are ignored by default and regenerated per run.
- Commit-facing portfolio artifacts are under `docs/portfolio/`.

## Visualization Results

### Throughput
![Throughput](docs/portfolio/figures/performance_tps.png)

### QoR Resources
![QoR](docs/portfolio/figures/qor_resources.png)

### ONNX MAE
![ONNX MAE](docs/portfolio/figures/onnx_mae.png)
