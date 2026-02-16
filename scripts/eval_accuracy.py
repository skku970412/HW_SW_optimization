#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.golden.golden_attention import scaled_dot_product_attention, softmax_approx, softmax_exact
from tests.golden.golden_ops import clamp_int8, clamp_int16


@dataclass
class AccuracyMetrics:
    softmax_mae: float
    softmax_max_abs: float
    attention_mae: float
    attention_max_abs: float
    quant_gemm_rel_l2: float


def _softmax_metrics(rng: np.random.Generator, cases: int, dim: int) -> tuple[float, float]:
    mae = []
    maxe = []
    for _ in range(cases):
        x = rng.normal(size=(dim, dim)).astype(np.float32)
        ex = softmax_exact(x, axis=-1)
        ap = softmax_approx(x, axis=-1)
        err = np.abs(ex - ap)
        mae.append(float(np.mean(err)))
        maxe.append(float(np.max(err)))
    return float(np.mean(mae)), float(np.max(maxe))


def _attention_metrics(rng: np.random.Generator, cases: int, seq: int, dim: int) -> tuple[float, float]:
    mae = []
    maxe = []
    for _ in range(cases):
        q = rng.normal(size=(seq, dim)).astype(np.float32)
        k = rng.normal(size=(seq, dim)).astype(np.float32)
        v = rng.normal(size=(seq, dim)).astype(np.float32)
        out_exact, _ = scaled_dot_product_attention(q, k, v, causal=True, use_approx_softmax=False)
        out_aprx, _ = scaled_dot_product_attention(q, k, v, causal=True, use_approx_softmax=True)
        err = np.abs(out_exact - out_aprx)
        mae.append(float(np.mean(err)))
        maxe.append(float(np.max(err)))
    return float(np.mean(mae)), float(np.max(maxe))


def _quant_gemm_metric(rng: np.random.Generator, cases: int, m: int, k: int, n: int) -> float:
    rel_l2 = []
    for _ in range(cases):
        a_f = rng.normal(scale=0.5, size=(m, k)).astype(np.float32)
        b_f = rng.normal(scale=0.5, size=(k, n)).astype(np.float32)
        y_f = a_f @ b_f

        a_q = clamp_int16(np.round(a_f * 128.0))
        b_q = clamp_int8(np.round(b_f * 64.0))
        y_q = (a_q.astype(np.int32) @ b_q.astype(np.int32)).astype(np.float32)
        y_dq = y_q / (128.0 * 64.0)

        num = np.linalg.norm(y_f - y_dq)
        den = np.linalg.norm(y_f) + 1e-9
        rel_l2.append(float(num / den))
    return float(np.mean(rel_l2))


def run_eval(seed: int, cases: int) -> AccuracyMetrics:
    rng = np.random.default_rng(seed)
    softmax_mae, softmax_max_abs = _softmax_metrics(rng, cases=cases, dim=16)
    attention_mae, attention_max_abs = _attention_metrics(rng, cases=cases, seq=16, dim=16)
    quant_gemm_rel_l2 = _quant_gemm_metric(rng, cases=cases, m=8, k=16, n=8)
    return AccuracyMetrics(
        softmax_mae=softmax_mae,
        softmax_max_abs=softmax_max_abs,
        attention_mae=attention_mae,
        attention_max_abs=attention_max_abs,
        quant_gemm_rel_l2=quant_gemm_rel_l2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate boardless accuracy metrics.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=Path("results/accuracy_report.json"))
    parser.add_argument("--csv-out", type=Path, default=Path("results/accuracy_report.csv"))
    args = parser.parse_args()

    metrics = run_eval(seed=args.seed, cases=args.cases)
    data = asdict(metrics)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with args.csv_out.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["metric", "value"])
        for k, v in data.items():
            w.writerow([k, f"{v:.8f}"])

    print(f"accuracy written: {args.json_out}, {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
