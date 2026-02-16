from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _has_iverilog() -> bool:
    if shutil.which("iverilog") and shutil.which("vvp"):
        return True
    return Path(r"C:\iverilog\bin\iverilog.exe").exists() and Path(r"C:\iverilog\bin\vvp.exe").exists()


@pytest.mark.skipif(not _has_iverilog(), reason="iverilog/vvp not found")
def test_cycle_model_calibration_script():
    subprocess.run(
        [
            "python",
            "scripts/calibrate_cycle_model.py",
            "--k-tiles",
            "2,4,8,16",
            "--prompt-len",
            "4",
            "--gen-len",
            "6",
            "--dim",
            "16",
        ],
        cwd=ROOT,
        check=True,
    )

    csv_path = ROOT / "results" / "model_calibration.csv"
    json_path = ROOT / "results" / "model_calibration.json"
    md_path = ROOT / "results" / "model_calibration.md"

    assert csv_path.exists()
    assert json_path.exists()
    assert md_path.exists()

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 4
    assert "predicted_cycles_per_token_calibrated" in rows[0]

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "scale" in payload
    assert "bias" in payload
    assert "mae_raw" in payload
    assert "mae_calibrated" in payload
    assert float(payload["mae_calibrated"]) <= float(payload["mae_raw"]) + 1e-9
