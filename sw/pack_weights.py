#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack int8 weights into a flat binary file.")
    parser.add_argument("--indir", type=Path, default=Path("sw/artifacts/tiny_decoder"))
    parser.add_argument("--outdir", type=Path, default=Path("sw/artifacts/tiny_decoder_packed"))
    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((indir / "meta.json").read_text(encoding="utf-8"))
    w_q = np.load(indir / "w_q_int8.npy")
    w_k = np.load(indir / "w_k_int8.npy")
    w_v = np.load(indir / "w_v_int8.npy")

    packed = np.concatenate(
        [
            w_q.reshape(-1).astype(np.int8),
            w_k.reshape(-1).astype(np.int8),
            w_v.reshape(-1).astype(np.int8),
        ]
    )
    (outdir / "weights_int8.bin").write_bytes(packed.tobytes())

    # Runtime currently consumes npy + meta for readability.
    np.save(outdir / "w_q_int8.npy", w_q)
    np.save(outdir / "w_k_int8.npy", w_k)
    np.save(outdir / "w_v_int8.npy", w_v)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"packed {packed.size} int8 values into {outdir / 'weights_int8.bin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
