# Transformer Acceleration (Boardless LLM Inference)
Repository for boardless development, validation, and portfolio packaging
of an FPGA/NPU-style LLM inference accelerator.
## TL;DR
- Same-scale proxy primary KPI: `fpga_est_tps / scaleup_proxy_tps = 1.804x`
- Boardless flow includes DSE, QoR, calibration, and one-command reproduction.
- Current numbers are pre-silicon proxy/estimate values; board measurement is the next step.
## Current Status
- Boardless track: B1~B8 PASS
- Optimization round: N1~N13 PASS
- tiny_cpu_tps: 6383.488061
- fpga_est_tps: 171.326754
- scaleup_proxy_tps: 94.960900
- speedup_fpga_est_vs_scaleup_proxy: 1.804182 (primary)
- onnx_mae_avg: 0.047009
- dse_best(k_tile/pe/overhead): 8/128/8
- cycle_model_calibration_improvement_pct: 92.70
## Metric Interpretation
- `tiny_cpu_tps`: throughput of dim=16 tiny regression path (reference-only metric).
- `fpga_est_tps`: estimated throughput from cycle model + QoR on distilgpt2-proxy scale.
- `scaleup_proxy_tps`: NumPy runtime throughput on distilgpt2-proxy scale.
- `fpga_est_tps / scaleup_proxy_tps`: primary KPI for fair same-scale comparison.
## Claims and Evidence
| claim | evidence | reproduce |
|---|---|---|
| Same-scale primary KPI speedup | `results/benchmark_suite.csv`, `docs/portfolio/final_report.md` | `powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1` |
| Cycle model calibration applied | `results/model_calibration.csv`, `results/model_calibration.json`, `docs/portfolio/figures/cycle_calibration.png` | `powershell -ExecutionPolicy Bypass -File scripts/run_n9_calibration.ps1 -MaxRuns 10` |
| DSE + Pareto optimization trace | `results/dse_autotune.csv`, `results/dse_pareto.csv`, `docs/portfolio/figures/dse_pareto.png` | `powershell -ExecutionPolicy Bypass -File scripts/run_n8_dse.ps1 -MaxRuns 10` |
| kv_cache BRAM mapping enabled | `results/qor_summary.csv` (`kv_cache` row, `bram > 0`) | `powershell -ExecutionPolicy Bypass -File scripts/run_vivado_qor.ps1` |
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
### Throughput (Primary KPI)
![Throughput](docs/portfolio/figures/performance_tps.png)
### Throughput (Reference, includes Tiny CPU)
![Throughput All](docs/portfolio/figures/performance_all_tps.png)
### QoR Resources
![QoR](docs/portfolio/figures/qor_resources.png)
### ONNX MAE
![ONNX MAE](docs/portfolio/figures/onnx_mae.png)
### DSE Top-5 (if available)
![DSE Top5](docs/portfolio/figures/dse_top5_cycles.png)
### DSE Pareto (if available)
![DSE Pareto](docs/portfolio/figures/dse_pareto.png)
### Cycle Model Calibration (if available)
![Calibration](docs/portfolio/figures/cycle_calibration.png)
