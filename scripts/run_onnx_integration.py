#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.api import BoardlessNpuRuntime, RuntimeConfig
from runtime.np_kernels import gemm_int8w_int16a_acc32


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    onnx_dir = ROOT / "sw" / "artifacts" / "onnx_proxy"
    pack_dir = ROOT / "sw" / "artifacts" / "onnx_proxy_packed"
    _run(["python", "sw/export_proxy_onnx.py", "--dim", "16", "--seed", "7", "--outdir", str(onnx_dir)])
    _run(["python", "sw/onnx_to_pack.py", "--onnx", str(onnx_dir / "tiny_decoder.onnx"), "--outdir", str(pack_dir)])

    sess = ort.InferenceSession(str(onnx_dir / "tiny_decoder.onnx"), providers=["CPUExecutionProvider"])
    x = np.ones((1, 16), dtype=np.float32)
    q_ref, k_ref, v_ref = sess.run(["Q", "K", "V"], {"X": x})

    w_q_i8 = np.load(pack_dir / "w_q_int8.npy")
    w_k_i8 = np.load(pack_dir / "w_k_int8.npy")
    w_v_i8 = np.load(pack_dir / "w_v_int8.npy")
    meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    scale = float(meta["dequant_scale"])

    x_i16 = np.round(x * 128.0).astype(np.int16)
    q_q = gemm_int8w_int16a_acc32(x_i16, w_q_i8).astype(np.float32) * (scale / 128.0)
    k_q = gemm_int8w_int16a_acc32(x_i16, w_k_i8).astype(np.float32) * (scale / 128.0)
    v_q = gemm_int8w_int16a_acc32(x_i16, w_v_i8).astype(np.float32) * (scale / 128.0)

    mae_q = float(np.mean(np.abs(q_ref - q_q)))
    mae_k = float(np.mean(np.abs(k_ref - k_q)))
    mae_v = float(np.mean(np.abs(v_ref - v_q)))

    rt = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=128))
    rt.init()
    rt.load(pack_dir)
    out = rt.run(prompt_tokens=np.ones((4, 16), dtype=np.int16), gen_len=4)
    st = rt.poll()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "onnx_path": str(onnx_dir / "tiny_decoder.onnx"),
        "mae_q": mae_q,
        "mae_k": mae_k,
        "mae_v": mae_v,
        "runtime_output_shape": list(out.shape),
        "runtime_status": st,
    }
    out_json = ROOT / "results" / "onnx_integration_result.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"onnx integration done: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
