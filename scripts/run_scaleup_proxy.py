#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.api import BoardlessNpuRuntime, RuntimeConfig


def main() -> int:
    dim = 768
    asset = ROOT / "sw" / "artifacts" / "distilgpt2_proxy"
    packed = ROOT / "sw" / "artifacts" / "distilgpt2_proxy_packed"

    subprocess.run(
        ["python", "sw/create_tiny_decoder_assets.py", "--dim", str(dim), "--seed", "42", "--outdir", str(asset)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["python", "sw/pack_weights.py", "--indir", str(asset), "--outdir", str(packed)],
        cwd=ROOT,
        check=True,
    )

    rt = BoardlessNpuRuntime(RuntimeConfig(dim=dim, max_seq=256))
    rt.init()
    rt.load(packed)

    prompt = np.ones((8, dim), dtype=np.int16)
    t0 = time.perf_counter()
    out = rt.run(prompt_tokens=prompt, gen_len=2)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    status = rt.poll()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dim": dim,
        "prompt_shape": list(prompt.shape),
        "output_shape": list(out.shape),
        "elapsed_sec": elapsed,
        "status": status,
    }
    out_json = ROOT / "results" / "scaleup_proxy_result.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"scale-up done: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
