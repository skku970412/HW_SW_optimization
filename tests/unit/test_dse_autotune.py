from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dse_autotune_script():
    subprocess.run(
        [
            "python",
            "scripts/run_dse_autotune.py",
            "--dim",
            "16",
            "--prompt-len",
            "6",
            "--gen-len",
            "6",
            "--k-tiles",
            "4,8",
            "--pe-macs",
            "64,128",
            "--overheads",
            "8,12",
        ],
        cwd=ROOT,
        check=True,
    )

    csv_path = ROOT / "results" / "dse_autotune.csv"
    pareto_path = ROOT / "results" / "dse_pareto.csv"
    best_path = ROOT / "results" / "dse_autotune_best.json"
    md_path = ROOT / "results" / "dse_autotune.md"

    assert csv_path.exists()
    assert pareto_path.exists()
    assert best_path.exists()
    assert md_path.exists()

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 8
    assert "tps_est_at_clock" in rows[0]
    assert "score_tps_per_area" in rows[0]
    assert "area_proxy" in rows[0]

    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert best["num_trials"] == 8
    assert best["pareto_count"] >= 1
    assert float(best["best"]["tps_est_at_clock"]) > 0.0
