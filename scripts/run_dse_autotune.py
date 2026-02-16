#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from runtime.register_map import STATUS_DONE


def _parse_list(text: str) -> list[int]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("empty search list")
    return out


def _ensure_assets(dim: int) -> Path:
    asset = ROOT / "sw" / "artifacts" / f"tiny_decoder_dse_d{dim}"
    packed = ROOT / "sw" / "artifacts" / f"tiny_decoder_dse_d{dim}_packed"
    subprocess.run(
        ["python", "sw/create_tiny_decoder_assets.py", "--dim", str(dim), "--seed", "11", "--outdir", str(asset)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["python", "sw/pack_weights.py", "--indir", str(asset), "--outdir", str(packed)],
        cwd=ROOT,
        check=True,
    )
    return packed


def _run_one(
    *,
    pack_dir: Path,
    dim: int,
    prompt_len: int,
    gen_len: int,
    cfg_k_tile: int,
    pe_mac_per_cycle: int,
    token_overhead_cycles: int,
    clock_mhz: float,
) -> dict[str, float | int | str]:
    rt = BoardlessNpuRuntime(
        RuntimeConfig(
            dim=dim,
            max_seq=256,
            backend="rtl",
            cfg_k_tile=cfg_k_tile,
            pe_mac_per_cycle=pe_mac_per_cycle,
            token_overhead_cycles=token_overhead_cycles,
        )
    )
    rt.init()
    rt.load(pack_dir)

    prompt = np.ones((prompt_len, dim), dtype=np.int16)
    out = rt.run(prompt_tokens=prompt, gen_len=gen_len)
    st = rt.poll()

    perf_cycles = int(st["perf_cycles"])
    perf_tokens = int(st["perf_tokens"])
    cycles_per_token = (perf_cycles / perf_tokens) if perf_tokens > 0 else 0.0
    tps_est = (clock_mhz * 1_000_000.0 / cycles_per_token) if cycles_per_token > 0 else 0.0
    # Simple area/energy proxy for constrained DSE ranking.
    area_proxy = float(pe_mac_per_cycle) + float(cfg_k_tile) * 8.0 + float(token_overhead_cycles) * 2.0
    score_tps_per_area = (tps_est / area_proxy) if area_proxy > 0 else 0.0
    edp_proxy = cycles_per_token * area_proxy

    return {
        "status": "PASS" if int(st["status"]) == STATUS_DONE else "FAIL",
        "cfg_k_tile": int(cfg_k_tile),
        "pe_mac_per_cycle": int(pe_mac_per_cycle),
        "token_overhead_cycles": int(token_overhead_cycles),
        "prompt_len": int(prompt_len),
        "gen_len": int(gen_len),
        "perf_cycles": perf_cycles,
        "perf_tokens": perf_tokens,
        "perf_stall_in": int(st["perf_stall_in"]),
        "perf_stall_out": int(st["perf_stall_out"]),
        "cycles_per_token": float(cycles_per_token),
        "tps_est_at_clock": float(tps_est),
        "area_proxy": float(area_proxy),
        "score_tps_per_area": float(score_tps_per_area),
        "edp_proxy": float(edp_proxy),
        "output_shape": f"{out.shape[0]}x{out.shape[1]}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run boardless RTL-backend DSE autotuning sweep.")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--gen-len", type=int, default=8)
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    parser.add_argument("--k-tiles", default="4,8,16")
    parser.add_argument("--pe-macs", default="64,128,256")
    parser.add_argument("--overheads", default="8,12,16")
    parser.add_argument("--out-csv", type=Path, default=Path("results/dse_autotune.csv"))
    parser.add_argument("--out-pareto", type=Path, default=Path("results/dse_pareto.csv"))
    parser.add_argument("--out-best", type=Path, default=Path("results/dse_autotune_best.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/dse_autotune.md"))
    args = parser.parse_args()

    k_tiles = _parse_list(args.k_tiles)
    pe_macs = _parse_list(args.pe_macs)
    overheads = _parse_list(args.overheads)
    pack_dir = _ensure_assets(args.dim)

    rows: list[dict[str, float | int | str]] = []
    run_id = 0
    for k_tile in k_tiles:
        for pe_mac in pe_macs:
            for overhead in overheads:
                run_id += 1
                row = _run_one(
                    pack_dir=pack_dir,
                    dim=args.dim,
                    prompt_len=args.prompt_len,
                    gen_len=args.gen_len,
                    cfg_k_tile=k_tile,
                    pe_mac_per_cycle=pe_mac,
                    token_overhead_cycles=overhead,
                    clock_mhz=args.clock_mhz,
                )
                row["run_id"] = run_id
                rows.append(row)

    rows.sort(key=lambda r: float(r["score_tps_per_area"]), reverse=True)
    best = rows[0] if rows else {}

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "run_id",
            "status",
            "cfg_k_tile",
            "pe_mac_per_cycle",
            "token_overhead_cycles",
            "prompt_len",
            "gen_len",
            "perf_cycles",
            "perf_tokens",
            "perf_stall_in",
            "perf_stall_out",
            "cycles_per_token",
            "tps_est_at_clock",
            "area_proxy",
            "score_tps_per_area",
            "edp_proxy",
            "output_shape",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Pareto frontier: maximize tps_est_at_clock, minimize area_proxy.
    pareto: list[dict[str, float | int | str]] = []
    pass_rows = [r for r in rows if r["status"] == "PASS"]
    pass_rows.sort(key=lambda r: float(r["tps_est_at_clock"]), reverse=True)
    best_area = float("inf")
    for row in pass_rows:
        area = float(row["area_proxy"])
        if area < best_area:
            pareto.append(row)
            best_area = area

    with args.out_pareto.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "run_id",
            "cfg_k_tile",
            "pe_mac_per_cycle",
            "token_overhead_cycles",
            "cycles_per_token",
            "tps_est_at_clock",
            "area_proxy",
            "score_tps_per_area",
            "edp_proxy",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in pareto:
            writer.writerow({k: row[k] for k in fieldnames})

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dim": args.dim,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "clock_mhz": args.clock_mhz,
        "num_trials": len(rows),
        "pareto_count": len(pareto),
        "best": best,
    }
    args.out_best.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_n = rows[:5]
    lines = [
        "# DSE Autotune Summary",
        "",
        f"- dim: {args.dim}",
        f"- prompt_len: {args.prompt_len}",
        f"- gen_len: {args.gen_len}",
        f"- clock_mhz: {args.clock_mhz}",
        f"- trials: {len(rows)}",
        f"- pareto_points: {len(pareto)}",
        "",
        "## Best Config",
        "",
        f"- cfg_k_tile: {best.get('cfg_k_tile', 'n/a')}",
        f"- pe_mac_per_cycle: {best.get('pe_mac_per_cycle', 'n/a')}",
        f"- token_overhead_cycles: {best.get('token_overhead_cycles', 'n/a')}",
        f"- cycles_per_token: {best.get('cycles_per_token', 'n/a')}",
        f"- tps_est_at_clock: {best.get('tps_est_at_clock', 'n/a')}",
        f"- area_proxy: {best.get('area_proxy', 'n/a')}",
        f"- score_tps_per_area: {best.get('score_tps_per_area', 'n/a')}",
        "",
        "## Top 5 Trials",
        "",
        "| rank | k_tile | pe_mac | overhead | cycles/token | tps_est | area | score | stall_in |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(top_n, start=1):
        lines.append(
            f"| {i} | {row['cfg_k_tile']} | {row['pe_mac_per_cycle']} | "
            f"{row['token_overhead_cycles']} | {float(row['cycles_per_token']):.3f} | "
            f"{float(row['tps_est_at_clock']):.3f} | {float(row['area_proxy']):.3f} | "
            f"{float(row['score_tps_per_area']):.3f} | {row['perf_stall_in']} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"dse autotune done: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
