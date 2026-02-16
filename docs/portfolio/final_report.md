# Final Portfolio Report

- generated_utc: 2026-02-16T10:57:23Z
- benchmark_suite_timestamp_utc: 2026-02-16T10:57:18Z
- scope: Boardless LLM inference accelerator MVP + optimization round (N1~N6)

## KPI Summary

| KPI | Value |
|---|---:|
| tiny_cpu_tps | 7212.405362 |
| fpga_est_tps | 171.326754 |
| scaleup_proxy_tps | 108.536976 |
| speedup_fpga_est_vs_tiny_cpu | 0.023754 |
| tiny_cpu_ms_per_token | 0.138650 |
| fpga_est_ms_per_token | 5.836800 |
| onnx_mae_avg | 0.047009 |
| qor_best_wns_ns | 3.682000 |

## Figures

### Throughput
![Throughput](figures/performance_tps.png)

### QoR Resources
![QoR](figures/qor_resources.png)

### ONNX MAE
![ONNX MAE](figures/onnx_mae.png)

## QoR Table

| top | lut | ff | dsp | bram | uram | wns_ns |
|---|---:|---:|---:|---:|---:|---:|
| attention_core | 46 | 32 | 3 | 0 | 0 | 3.682 |
| decoder_block_top | 46 | 32 | 3 | 0 | 0 | 3.682 |
| gemm_core | 26 | 40 | 1 | 0 | 0 | 2.940 |
| kv_cache | 2513 | 8224 | 0 | 0 | 0 | 3.341 |

## Validation Policy

1. Run each validation step up to 10 times.
2. Stop early when PASS is reached once.
3. Record PASS/FAIL/BLOCKED in logs.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1
```
