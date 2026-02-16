from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_accuracy_eval_script_and_thresholds():
    json_out = ROOT / "results" / "accuracy_report.json"
    csv_out = ROOT / "results" / "accuracy_report.csv"
    subprocess.run(
        [
            "python",
            "scripts/eval_accuracy.py",
            "--seed",
            "2026",
            "--cases",
            "8",
            "--json-out",
            str(json_out),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["softmax_mae"] < 0.06
    assert data["softmax_max_abs"] < 0.25
    assert data["attention_mae"] < 0.20
    assert data["attention_max_abs"] < 1.00
    assert data["quant_gemm_rel_l2"] < 0.20
