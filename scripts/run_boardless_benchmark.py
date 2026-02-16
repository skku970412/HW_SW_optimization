#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from scripts.perf_model import PerfInput, estimate


def _ensure_flow_assets() -> None:
    subprocess.run(["python", "scripts/run_sw_hw_flow.py"], cwd=ROOT, check=True)


def _measure_cpu_runtime(prompt_len: int, gen_len: int, warmup: int, repeats: int) -> tuple[float, float]:
    rt = BoardlessNpuRuntime(RuntimeConfig(dim=16, max_seq=256))
    rt.init()
    rt.load(ROOT / "sw/artifacts/tiny_decoder_packed")

    prompt = np.ones((prompt_len, 16), dtype=np.int16)

    for _ in range(warmup):
        _ = rt.run(prompt_tokens=prompt, gen_len=gen_len)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = rt.run(prompt_tokens=prompt, gen_len=gen_len)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_s = float(np.mean(times))
    throughput = gen_len / avg_s
    latency_ms = (avg_s / gen_len) * 1000.0
    return latency_ms, throughput


def main() -> int:
    parser = argparse.ArgumentParser(description="Run boardless benchmark and export CSV.")
    parser.add_argument("--out", type=Path, default=Path("results/benchmark_actual.csv"))
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-power-w", type=float, default=65.0, help="Assumed CPU package power for energy/token estimate")
    args = parser.parse_args()

    _ensure_flow_assets()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    latency_ms, throughput = _measure_cpu_runtime(
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    energy_per_token = args.cpu_power_w / throughput if throughput > 0 else float("nan")

    perf = estimate(
        PerfInput(
            layers=6,
            hidden=768,
            seq=256,
            pe_mac_per_cycle=256,
            clock_mhz=200.0,
            efficiency=0.15,
        )
    )
    fpga_tps = perf.effective_tokens_per_sec
    fpga_latency = 1000.0 / fpga_tps if fpga_tps > 0 else float("nan")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "run_id",
                "date_utc",
                "model",
                "precision",
                "device",
                "board",
                "toolchain",
                "prompt_len",
                "gen_len",
                "batch_size",
                "latency_per_token_ms",
                "throughput_tokens_per_sec",
                "avg_power_w",
                "energy_per_token_j",
                "quality_metric",
                "quality_value",
                "quality_drop_pct",
                "notes",
            ]
        )
        writer.writerow(
            [
                "boardless_cpu_runtime_001",
                ts,
                "tiny_decoder_dim16",
                "int8w_int16a",
                "cpu",
                "na",
                "numpy_runtime_api",
                args.prompt_len,
                args.gen_len,
                1,
                f"{latency_ms:.6f}",
                f"{throughput:.6f}",
                f"{args.cpu_power_w:.2f}",
                f"{energy_per_token:.6f}",
                "na",
                "",
                "",
                "Measured in boardless runtime on host CPU",
            ]
        )
        writer.writerow(
            [
                "boardless_fpga_est_001",
                ts,
                "distilgpt2_proxy",
                "int8w_int16a",
                "fpga_estimated",
                "kv260",
                "perf_model",
                256,
                32,
                1,
                f"{fpga_latency:.6f}",
                f"{fpga_tps:.6f}",
                "",
                "",
                "na",
                "",
                "",
                "Estimated from scripts/perf_model.py reference formula",
            ]
        )

    md = args.out.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                "# Boardless Benchmark",
                "",
                f"- output_csv: `{args.out}`",
                f"- prompt_len: {args.prompt_len}",
                f"- gen_len: {args.gen_len}",
                f"- warmup: {args.warmup}",
                f"- repeats: {args.repeats}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"benchmark written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
