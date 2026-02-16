from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_boardless_benchmark_script():
    out = ROOT / "results" / "benchmark_actual.csv"
    subprocess.run(
        [
            "python",
            "scripts/run_boardless_benchmark.py",
            "--out",
            str(out),
            "--prompt-len",
            "16",
            "--gen-len",
            "8",
            "--warmup",
            "1",
            "--repeats",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    assert out.exists()

    with out.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) >= 2
    assert rows[0]["device"] == "cpu"
    assert rows[1]["device"] == "fpga_estimated"
