#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    _run(
        [
            "python",
            "scripts/run_boardless_benchmark.py",
            "--out",
            "results/benchmark_actual.csv",
            "--prompt-len",
            "16",
            "--gen-len",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "3",
        ]
    )
    _run(["python", "scripts/run_scaleup_proxy.py"])
    _run(["python", "scripts/run_onnx_integration.py"])

    rows = list(csv.DictReader((ROOT / "results/benchmark_actual.csv").open("r", encoding="utf-8", newline="")))
    cpu_row = next(r for r in rows if r["device"] == "cpu")
    fpga_row = next(r for r in rows if r["device"] == "fpga_estimated")

    scale = json.loads((ROOT / "results/scaleup_proxy_result.json").read_text(encoding="utf-8"))
    onnx = json.loads((ROOT / "results/onnx_integration_result.json").read_text(encoding="utf-8"))

    scale_tps = 2.0 / float(scale["elapsed_sec"]) if float(scale["elapsed_sec"]) > 0 else 0.0
    tiny_cpu_tps = float(cpu_row["throughput_tokens_per_sec"])
    fpga_est_tps = float(fpga_row["throughput_tokens_per_sec"])
    speedup_fpga_vs_tiny_cpu = fpga_est_tps / tiny_cpu_tps if tiny_cpu_tps > 0 else 0.0

    suite_csv = ROOT / "results/benchmark_suite.csv"
    with suite_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["item", "value"])
        w.writerow(["timestamp_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])
        w.writerow(["tiny_cpu_tps", f"{tiny_cpu_tps:.6f}"])
        w.writerow(["fpga_est_tps", f"{fpga_est_tps:.6f}"])
        w.writerow(["scaleup_proxy_tps", f"{scale_tps:.6f}"])
        w.writerow(["speedup_fpga_est_vs_tiny_cpu", f"{speedup_fpga_vs_tiny_cpu:.6f}"])
        w.writerow(["onnx_mae_q", f"{onnx['mae_q']:.6f}"])
        w.writerow(["onnx_mae_k", f"{onnx['mae_k']:.6f}"])
        w.writerow(["onnx_mae_v", f"{onnx['mae_v']:.6f}"])

    summary_md = ROOT / "results/benchmark_suite.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Benchmark Suite Summary",
                "",
                f"- tiny_cpu_tps: {tiny_cpu_tps:.6f}",
                f"- fpga_est_tps: {fpga_est_tps:.6f}",
                f"- scaleup_proxy_tps: {scale_tps:.6f}",
                f"- speedup_fpga_est_vs_tiny_cpu: {speedup_fpga_vs_tiny_cpu:.6f}",
                f"- onnx_mae_q/k/v: {onnx['mae_q']:.6f} / {onnx['mae_k']:.6f} / {onnx['mae_v']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"benchmark suite done: {suite_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
