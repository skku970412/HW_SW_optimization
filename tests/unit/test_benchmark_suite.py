from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_suite_script():
    subprocess.run(["python", "scripts/run_benchmark_suite.py"], cwd=ROOT, check=True)
    p = ROOT / "results" / "benchmark_suite.csv"
    assert p.exists()

    rows = list(csv.DictReader(p.open("r", encoding="utf-8", newline="")))
    keys = {r["item"] for r in rows}
    for req in [
        "tiny_cpu_tps",
        "fpga_est_tps",
        "scaleup_proxy_tps",
        "speedup_fpga_est_vs_tiny_cpu",
        "onnx_mae_q",
        "onnx_mae_k",
        "onnx_mae_v",
    ]:
        assert req in keys
