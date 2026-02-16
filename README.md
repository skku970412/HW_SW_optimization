# Transformer Acceleration (Boardless LLM Inference)
Repository for boardless development, validation, and portfolio packaging
of an FPGA/NPU-style LLM inference accelerator.
## Current Status
- Boardless track: B1~B8 PASS
- Optimization round: N1~N10 PASS
- tiny_cpu_tps: 9294.760020
- fpga_est_tps: 171.326754
- scaleup_proxy_tps: 99.066791
- speedup_fpga_est_vs_scaleup_proxy: 1.729407 (primary)
- onnx_mae_avg: 0.047009
- dse_best(k_tile/pe/overhead): 8/128/8
- cycle_model_calibration_improvement_pct: 92.70
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
## P2 DSE/Autotune
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_n8_dse.ps1 -MaxRuns 10
python scripts/run_dse_autotune.py
```
## N9 Cycle-model Calibration
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_n9_calibration.ps1 -MaxRuns 10
python scripts/calibrate_cycle_model.py
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
### DSE Top-5 (if available)
![DSE Top5](docs/portfolio/figures/dse_top5_cycles.png)
### Cycle Model Calibration (if available)
![Calibration](docs/portfolio/figures/cycle_calibration.png)
