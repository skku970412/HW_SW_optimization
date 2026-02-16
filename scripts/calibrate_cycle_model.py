#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.api import BoardlessNpuRuntime, RuntimeConfig


def _find_bin(name: str, fallback: Path | None = None) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if fallback and fallback.exists():
        return str(fallback)
    return None


def _parse_list(text: str) -> list[int]:
    out: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("empty list")
    return out


@dataclass
class Obs:
    cfg_k_tile: int
    perf_cycles: int
    perf_tokens: int
    cycles_per_token: float


def _run_npu_top_obs(
    *,
    cfg_k_tile: int,
    prompt_len: int,
    gen_len: int,
    workdir: Path,
    iverilog: str,
    vvp: str,
) -> Obs:
    tb = workdir / f"tb_npu_top_k{cfg_k_tile}.sv"
    out = workdir / f"tb_npu_top_k{cfg_k_tile}.out"
    tb.write_text(
        "\n".join(
            [
                "`timescale 1ns/1ps",
                "module tb;",
                "  logic clk = 0;",
                "  logic rst_n = 0;",
                "  logic mmio_wr_en = 0;",
                "  logic mmio_rd_en = 0;",
                "  logic [7:0] mmio_addr = 0;",
                "  logic [31:0] mmio_wdata = 0;",
                "  logic [31:0] mmio_rdata;",
                "  logic mmio_ready;",
                "  npu_top dut(",
                "    .clk(clk), .rst_n(rst_n),",
                "    .mmio_wr_en(mmio_wr_en), .mmio_rd_en(mmio_rd_en),",
                "    .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata),",
                "    .mmio_rdata(mmio_rdata), .mmio_ready(mmio_ready)",
                "  );",
                "  always #5 clk = ~clk;",
                "  task mmio_write(input [7:0] a, input [31:0] d);",
                "    begin",
                "      @(posedge clk);",
                "      mmio_addr <= a;",
                "      mmio_wdata <= d;",
                "      mmio_wr_en <= 1'b1;",
                "      mmio_rd_en <= 1'b0;",
                "      @(posedge clk);",
                "      mmio_wr_en <= 1'b0;",
                "      mmio_wdata <= 32'h0;",
                "    end",
                "  endtask",
                "  task mmio_read(input [7:0] a, output [31:0] d);",
                "    begin",
                "      @(posedge clk);",
                "      mmio_addr <= a;",
                "      mmio_rd_en <= 1'b1;",
                "      mmio_wr_en <= 1'b0;",
                "      @(posedge clk);",
                "      d = mmio_rdata;",
                "      mmio_rd_en <= 1'b0;",
                "    end",
                "  endtask",
                "  integer i;",
                "  reg [31:0] st;",
                "  reg [31:0] cyc;",
                "  reg [31:0] tok;",
                "  initial begin",
                "    repeat (2) @(posedge clk);",
                "    rst_n <= 1'b1;",
                f"    mmio_write(8'h08, 32'd{prompt_len});",
                f"    mmio_write(8'h0C, 32'd{gen_len});",
                f"    mmio_write(8'h28, 32'd{cfg_k_tile});",
                "    mmio_write(8'h00, 32'h1);",
                "    for (i = 0; i < 4096; i = i + 1) begin",
                "      mmio_read(8'h04, st);",
                "      if (st[1]) begin",
                "        mmio_read(8'h18, cyc);",
                "        mmio_read(8'h1C, tok);",
                "        $display(\"PERF_CYCLES=%0d\", cyc);",
                "        $display(\"PERF_TOKENS=%0d\", tok);",
                "        $finish;",
                "      end",
                "      if (st[2]) begin",
                "        $display(\"ERROR_STATUS=1\");",
                "        $finish;",
                "      end",
                "    end",
                "    $display(\"TIMEOUT=1\");",
                "    $finish;",
                "  end",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd_compile = [
        iverilog,
        "-g2012",
        "-s",
        "tb",
        "-o",
        str(out),
        str(ROOT / "hw/rtl/npu_top.sv"),
        str(tb),
    ]
    subprocess.run(cmd_compile, check=True, cwd=ROOT)
    run = subprocess.run([vvp, str(out)], cwd=ROOT, check=True, text=True, capture_output=True)
    text = run.stdout + "\n" + run.stderr

    cyc = None
    tok = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("PERF_CYCLES="):
            cyc = int(line.split("=", 1)[1])
        if line.startswith("PERF_TOKENS="):
            tok = int(line.split("=", 1)[1])
    if cyc is None or tok is None or tok <= 0:
        raise RuntimeError(f"failed to parse npu_top perf counters for cfg_k_tile={cfg_k_tile}\n{text}")
    return Obs(cfg_k_tile=cfg_k_tile, perf_cycles=cyc, perf_tokens=tok, cycles_per_token=cyc / tok)


def _ensure_assets(dim: int) -> Path:
    asset = ROOT / "sw" / "artifacts" / "tiny_decoder"
    packed = ROOT / "sw" / "artifacts" / "tiny_decoder_packed"
    subprocess.run(
        ["python", "sw/create_tiny_decoder_assets.py", "--dim", str(dim), "--seed", "123", "--outdir", str(asset)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["python", "sw/pack_weights.py", "--indir", str(asset), "--outdir", str(packed)],
        cwd=ROOT,
        check=True,
    )
    return packed


def _predict_cpt(*, cfg_k_tile: int, prompt_len: int, gen_len: int, dim: int, pack_dir: Path) -> float:
    # Use runtime RTL proxy path as cycle-model prediction (uncalibrated).
    rt = BoardlessNpuRuntime(
        RuntimeConfig(
            dim=dim,
            max_seq=256,
            backend="rtl",
            cfg_k_tile=cfg_k_tile,
            cycle_calib_scale=1.0,
            cycle_calib_bias=0.0,
        )
    )
    rt.init()
    rt.load(pack_dir)
    prompt = np.ones((prompt_len, dim), dtype=np.int16)
    _ = rt.run(prompt_tokens=prompt, gen_len=gen_len)
    st = rt.poll()
    perf_cycles = int(st["perf_cycles"])
    perf_tokens = int(st["perf_tokens"])
    return (perf_cycles / perf_tokens) if perf_tokens > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate cycle model against observed RTL simulation counters.")
    parser.add_argument("--k-tiles", default="2,4,8,16")
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--gen-len", type=int, default=6)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=Path("results/model_calibration.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("results/model_calibration.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/model_calibration.md"))
    args = parser.parse_args()

    iverilog = _find_bin("iverilog", Path(r"C:\iverilog\bin\iverilog.exe"))
    vvp = _find_bin("vvp", Path(r"C:\iverilog\bin\vvp.exe"))
    if not iverilog or not vvp:
        print("iverilog/vvp not found; calibration requires RTL simulation.")
        return 2

    k_tiles = _parse_list(args.k_tiles)
    workdir = ROOT / "results" / "calibration_tmp"
    workdir.mkdir(parents=True, exist_ok=True)
    pack_dir = _ensure_assets(args.dim)

    obs_rows: list[Obs] = []
    pred_raw: list[float] = []
    for k in k_tiles:
        obs = _run_npu_top_obs(
            cfg_k_tile=k,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            workdir=workdir,
            iverilog=iverilog,
            vvp=vvp,
        )
        obs_rows.append(obs)
        pred_raw.append(
            _predict_cpt(
                cfg_k_tile=k,
                prompt_len=args.prompt_len,
                gen_len=args.gen_len,
                dim=args.dim,
                pack_dir=pack_dir,
            )
        )

    y = np.array([r.cycles_per_token for r in obs_rows], dtype=np.float64)
    x = np.array(pred_raw, dtype=np.float64)
    if len(x) >= 2 and np.std(x) > 1e-12:
        scale, bias = np.polyfit(x, y, 1)
    else:
        scale, bias = 1.0, 0.0
    y_raw = x
    y_cal = x * scale + bias

    mae_raw = float(np.mean(np.abs(y - y_raw)))
    mae_cal = float(np.mean(np.abs(y - y_cal)))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "cfg_k_tile",
                "observed_cycles_per_token",
                "predicted_cycles_per_token_raw",
                "predicted_cycles_per_token_calibrated",
                "abs_err_raw",
                "abs_err_calibrated",
            ]
        )
        for i, r in enumerate(obs_rows):
            err_raw = abs(y[i] - y_raw[i])
            err_cal = abs(y[i] - y_cal[i])
            w.writerow(
                [
                    r.cfg_k_tile,
                    f"{y[i]:.6f}",
                    f"{y_raw[i]:.6f}",
                    f"{y_cal[i]:.6f}",
                    f"{err_raw:.6f}",
                    f"{err_cal:.6f}",
                ]
            )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dim": args.dim,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "k_tiles": k_tiles,
        "scale": float(scale),
        "bias": float(bias),
        "mae_raw": mae_raw,
        "mae_calibrated": mae_cal,
        "improvement_pct": ((mae_raw - mae_cal) / mae_raw * 100.0) if mae_raw > 0 else 0.0,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.out_md.write_text(
        "\n".join(
            [
                "# Cycle Model Calibration",
                "",
                f"- dim: {args.dim}",
                f"- prompt_len: {args.prompt_len}",
                f"- gen_len: {args.gen_len}",
                f"- k_tiles: {','.join(str(k) for k in k_tiles)}",
                f"- scale: {payload['scale']:.6f}",
                f"- bias: {payload['bias']:.6f}",
                f"- mae_raw: {mae_raw:.6f}",
                f"- mae_calibrated: {mae_cal:.6f}",
                f"- improvement_pct: {payload['improvement_pct']:.2f}",
                "",
                f"- csv: `{args.out_csv}`",
                f"- json: `{args.out_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"calibration done: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
