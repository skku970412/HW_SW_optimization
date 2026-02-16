#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.golden.golden_ops import clamp_int8, clamp_int16, gemm_int8w_int16a_acc32


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GEMM test vectors for boardless flow.")
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("tests/vectors"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    a = clamp_int16(rng.integers(-1024, 1024, size=(args.m, args.k)))
    b = clamp_int8(rng.integers(-64, 64, size=(args.k, args.n)))
    y = gemm_int8w_int16a_acc32(a, b)

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "gemm_a_int16.npy", a)
    np.save(args.outdir / "gemm_b_int8.npy", b)
    np.save(args.outdir / "gemm_out_int32.npy", y)

    print(f"Saved vectors to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
