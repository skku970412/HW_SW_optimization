#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(w)))
    scale = 127.0 / max_abs if max_abs > 0 else 1.0
    q = np.clip(np.round(w * scale), -128, 127).astype(np.int8)
    dequant_scale = 1.0 / scale
    return q, dequant_scale


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tiny decoder weights for boardless SW-HW flow.")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=Path, default=Path("sw/artifacts/tiny_decoder"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    w_q_f = rng.normal(loc=0.0, scale=0.5, size=(args.dim, args.dim)).astype(np.float32)
    w_k_f = rng.normal(loc=0.0, scale=0.5, size=(args.dim, args.dim)).astype(np.float32)
    w_v_f = rng.normal(loc=0.0, scale=0.5, size=(args.dim, args.dim)).astype(np.float32)

    w_q_i8, s_q = quantize_int8(w_q_f)
    w_k_i8, s_k = quantize_int8(w_k_f)
    w_v_i8, s_v = quantize_int8(w_v_f)

    # Keep one dequant scale for runtime simplicity.
    dequant_scale = float((s_q + s_k + s_v) / 3.0)

    np.save(outdir / "w_q_int8.npy", w_q_i8)
    np.save(outdir / "w_k_int8.npy", w_k_i8)
    np.save(outdir / "w_v_int8.npy", w_v_i8)

    meta = {
        "dim": args.dim,
        "seed": args.seed,
        "dequant_scale": dequant_scale,
        "format": "int8_weight_npy",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"created tiny decoder assets at {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
