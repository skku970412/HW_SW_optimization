from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_portfolio_packaging_outputs():
    subprocess.run(["python", "scripts/generate_portfolio_assets.py"], cwd=ROOT, check=True)

    report = ROOT / "docs" / "portfolio" / "final_report.md"
    fig_dir = ROOT / "docs" / "portfolio" / "figures"
    manifest = ROOT / "docs" / "portfolio" / "manifest.json"
    runbook = ROOT / "docs" / "portfolio" / "runbook.md"
    readme = ROOT / "README.md"

    assert report.exists()
    assert (fig_dir / "performance_tps.png").exists()
    assert (fig_dir / "qor_resources.png").exists()
    assert (fig_dir / "onnx_mae.png").exists()
    assert manifest.exists()
    assert runbook.exists()
    assert readme.exists()

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "final_report" in data
    assert "metrics" in data
    assert "artifacts" in data
    assert len(data["figures"]) == 3
    assert data["final_report"] == "docs/portfolio/final_report.md"
    assert all(not p.startswith("D:\\") and not p.startswith("C:\\") for p in data["figures"])
