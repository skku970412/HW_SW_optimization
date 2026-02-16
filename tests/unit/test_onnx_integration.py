from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_onnx_integration_flow():
    subprocess.run(["python", "scripts/run_onnx_integration.py"], cwd=ROOT, check=True)
    p = ROOT / "results" / "onnx_integration_result.json"
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["runtime_output_shape"] == [4, 16]
    assert d["runtime_status"]["done_tokens"] == 4
    assert d["mae_q"] < 1.0
    assert d["mae_k"] < 1.0
    assert d["mae_v"] < 1.0
