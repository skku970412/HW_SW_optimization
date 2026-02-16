from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from runtime.api import BoardlessNpuRuntime, RuntimeConfig
from runtime.register_map import STATUS_DONE


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_direct_api_smoke():
    # Ensure assets exist via scripts.
    subprocess.run(
        ["python", "sw/create_tiny_decoder_assets.py", "--outdir", "sw/artifacts/tiny_decoder"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "sw/pack_weights.py",
            "--indir",
            "sw/artifacts/tiny_decoder",
            "--outdir",
            "sw/artifacts/tiny_decoder_packed",
        ],
        cwd=ROOT,
        check=True,
    )

    rt = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=64))
    rt.init()
    rt.load(ROOT / "sw/artifacts/tiny_decoder_packed")

    prompt = np.ones((3, 16), dtype=np.int16)
    out = rt.run(prompt_tokens=prompt, gen_len=4)
    status = rt.poll()

    assert out.shape == (4, 16)
    assert status["status"] == STATUS_DONE
    assert status["done_tokens"] == 4


def test_sw_hw_flow_script_generates_json():
    subprocess.run(["python", "scripts/run_sw_hw_flow.py"], cwd=ROOT, check=True)
    p = ROOT / "results" / "sw_hw_flow_result.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["output_shape"] == [8, 16]
