from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_rtl_backend_flow_script():
    subprocess.run(["python", "scripts/run_rtl_backend_flow.py"], cwd=ROOT, check=True)
    p = ROOT / "results" / "rtl_backend_flow_result.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["backend"] == "rtl_proxy"
    assert d["output_shape"] == [8, 16]
    assert int(d["status"]["perf_cycles"]) > 0
