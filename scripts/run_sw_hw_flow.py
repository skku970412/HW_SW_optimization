#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.api import BoardlessNpuRuntime, RuntimeConfig


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    asset_dir = ROOT / "sw" / "artifacts" / "tiny_decoder"
    pack_dir = ROOT / "sw" / "artifacts" / "tiny_decoder_packed"

    _run(["python", "sw/create_tiny_decoder_assets.py", "--dim", "16", "--seed", "123", "--outdir", str(asset_dir)])
    _run(["python", "sw/pack_weights.py", "--indir", str(asset_dir), "--outdir", str(pack_dir)])

    rt = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=256))
    rt.init()
    rt.load(pack_dir)

    prompt = np.ones((4, 16), dtype=np.int16)
    out = rt.run(prompt_tokens=prompt, gen_len=8)
    status = rt.poll()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_shape": list(prompt.shape),
        "output_shape": list(out.shape),
        "status": status,
    }
    out_json = ROOT / "results" / "sw_hw_flow_result.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"flow done, wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
