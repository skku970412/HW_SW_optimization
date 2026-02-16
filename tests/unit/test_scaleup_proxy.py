from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_scaleup_proxy_flow():
    subprocess.run(["python", "scripts/run_scaleup_proxy.py"], cwd=ROOT, check=True)
    p = ROOT / "results" / "scaleup_proxy_result.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["dim"] == 768
    assert d["output_shape"] == [2, 768]
    assert d["status"]["done_tokens"] == 2
