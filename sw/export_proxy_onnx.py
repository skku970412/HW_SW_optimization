#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a tiny decoder projection ONNX model.")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=Path("sw/artifacts/onnx_proxy"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    w_q = rng.normal(scale=0.2, size=(args.dim, args.dim)).astype(np.float32)
    w_k = rng.normal(scale=0.2, size=(args.dim, args.dim)).astype(np.float32)
    w_v = rng.normal(scale=0.2, size=(args.dim, args.dim)).astype(np.float32)

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, args.dim])
    q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, args.dim])
    k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, args.dim])
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, args.dim])

    wq_init = helper.make_tensor("W_Q", TensorProto.FLOAT, [args.dim, args.dim], w_q.flatten().tolist())
    wk_init = helper.make_tensor("W_K", TensorProto.FLOAT, [args.dim, args.dim], w_k.flatten().tolist())
    wv_init = helper.make_tensor("W_V", TensorProto.FLOAT, [args.dim, args.dim], w_v.flatten().tolist())

    n_q = helper.make_node("MatMul", ["X", "W_Q"], ["Q"], name="MatMul_Q")
    n_k = helper.make_node("MatMul", ["X", "W_K"], ["K"], name="MatMul_K")
    n_v = helper.make_node("MatMul", ["X", "W_V"], ["V"], name="MatMul_V")

    graph = helper.make_graph(
        nodes=[n_q, n_k, n_v],
        name="TinyDecoderProjection",
        inputs=[x],
        outputs=[q, k, v],
        initializer=[wq_init, wk_init, wv_init],
    )
    model = helper.make_model(graph, producer_name="boardless-flow")
    model.ir_version = 10
    if model.opset_import:
        model.opset_import[0].version = 13
    onnx.checker.check_model(model)

    onnx_path = outdir / "tiny_decoder.onnx"
    onnx.save(model, onnx_path)

    np.save(outdir / "w_q_float.npy", w_q)
    np.save(outdir / "w_k_float.npy", w_k)
    np.save(outdir / "w_v_float.npy", w_v)
    (outdir / "meta.json").write_text(
        json.dumps({"dim": args.dim, "seed": args.seed, "opset": 13}, indent=2),
        encoding="utf-8",
    )
    print(f"exported onnx: {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
