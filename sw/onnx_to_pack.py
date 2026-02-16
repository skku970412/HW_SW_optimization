#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _quant_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(w)))
    scale = 127.0 / max_abs if max_abs > 0 else 1.0
    q = np.clip(np.round(w * scale), -128, 127).astype(np.int8)
    dequant = 1.0 / scale
    return q, dequant


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX initializers into boardless packed INT8 weights.")
    parser.add_argument("--onnx", type=Path, default=Path("sw/artifacts/onnx_proxy/tiny_decoder.onnx"))
    parser.add_argument("--outdir", type=Path, default=Path("sw/artifacts/onnx_proxy_packed"))
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    w_q = inits["W_Q"].astype(np.float32)
    w_k = inits["W_K"].astype(np.float32)
    w_v = inits["W_V"].astype(np.float32)

    w_q_i8, s_q = _quant_int8(w_q)
    w_k_i8, s_k = _quant_int8(w_k)
    w_v_i8, s_v = _quant_int8(w_v)

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "w_q_int8.npy", w_q_i8)
    np.save(out / "w_k_int8.npy", w_k_i8)
    np.save(out / "w_v_int8.npy", w_v_i8)

    dequant_scale = float((s_q + s_k + s_v) / 3.0)
    meta = {"dim": int(w_q.shape[0]), "dequant_scale": dequant_scale, "source": str(args.onnx)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    packed = np.concatenate([w_q_i8.reshape(-1), w_k_i8.reshape(-1), w_v_i8.reshape(-1)])
    (out / "weights_int8.bin").write_bytes(packed.tobytes())

    print(f"packed from onnx: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
