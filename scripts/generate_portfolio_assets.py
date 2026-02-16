#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUT_DIR = ROOT / "docs" / "portfolio"

REQUIRED_SUITE_KEYS = [
    "tiny_cpu_tps",
    "fpga_est_tps",
    "scaleup_proxy_tps",
    "onnx_mae_q",
    "onnx_mae_k",
    "onnx_mae_v",
]


def _read_kv_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark suite csv not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            item = (row.get("item") or "").strip()
            value = (row.get("value") or "").strip()
            if item:
                out[item] = value
    return out


def _read_qor(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"qor summary csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _read_optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _detect_optimization_round(progress_csv: Path) -> str:
    rows = _read_optional_csv_rows(progress_csv)
    max_n = 0
    for row in rows:
        week = (row.get("week") or "").strip().upper()
        status = (row.get("status") or "").strip().upper()
        if status != "PASS":
            continue
        m = re.fullmatch(r"N(\d+)", week)
        if not m:
            continue
        n = int(m.group(1))
        if n > max_n:
            max_n = n
    if max_n <= 0:
        return "N-round unknown"
    return f"N1~N{max_n}"


def _to_float(suite: dict[str, str], key: str) -> float:
    try:
        return float(suite[key])
    except KeyError as exc:
        raise KeyError(f"missing key in benchmark suite: {key}") from exc
    except ValueError as exc:
        raise ValueError(f"non-numeric benchmark suite value for {key}: {suite.get(key)!r}") from exc


def _validate_suite(suite: dict[str, str]) -> None:
    missing = [k for k in REQUIRED_SUITE_KEYS if k not in suite]
    if missing:
        raise KeyError(f"benchmark suite missing required keys: {', '.join(missing)}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _make_perf_plot(suite: dict[str, str], figures_dir: Path) -> Path:
    labels = ["Scale-up Proxy", "FPGA Est"]
    values = [
        _to_float(suite, "scaleup_proxy_tps"),
        _to_float(suite, "fpga_est_tps"),
    ]
    tiny_cpu_tps = _to_float(suite, "tiny_cpu_tps")
    speedup = (values[1] / values[0]) if values[0] > 0 else 0.0
    ymax = max(values) * 1.25 if values else 1.0

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values)
    plt.ylabel("Tokens/sec")
    plt.title("Primary Throughput KPI (Same-scale Proxy)")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylim(0, ymax)
    plt.text(
        0.5,
        ymax * 0.96,
        f"speedup_fpga_est_vs_scaleup_proxy = {speedup:.3f}x",
        ha="center",
        va="top",
        fontsize=9,
    )
    plt.text(
        0.98,
        0.02,
        f"Tiny CPU reference (different scale): {tiny_cpu_tps:.1f} tps",
        ha="right",
        va="bottom",
        fontsize=8,
        transform=plt.gca().transAxes,
    )
    out = figures_dir / "performance_tps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _make_perf_all_plot(suite: dict[str, str], figures_dir: Path) -> Path:
    labels = ["Tiny CPU", "Scale-up Proxy", "FPGA Est"]
    values = [
        _to_float(suite, "tiny_cpu_tps"),
        _to_float(suite, "scaleup_proxy_tps"),
        _to_float(suite, "fpga_est_tps"),
    ]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values)
    plt.ylabel("Tokens/sec")
    plt.title("Throughput Reference (includes Tiny CPU)")
    plt.yscale("log")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    out = figures_dir / "performance_all_tps.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _make_qor_plot(qor_rows: list[dict[str, str]], figures_dir: Path) -> Path:
    tops = [row["top"] for row in qor_rows]
    lut = [float(row["lut"] or 0.0) for row in qor_rows]
    ff = [float(row["ff"] or 0.0) for row in qor_rows]
    dsp = [float(row["dsp"] or 0.0) for row in qor_rows]
    bram = [float(row["bram"] or 0.0) for row in qor_rows]

    fig, (ax_main, ax_bram) = plt.subplots(
        2,
        1,
        figsize=(9, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = range(len(tops))
    width = 0.25
    ax_main.bar([i - width for i in x], lut, width=width, label="LUT")
    ax_main.bar(list(x), ff, width=width, label="FF")
    ax_main.bar([i + width for i in x], dsp, width=width, label="DSP")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Count (log scale)")
    ax_main.set_title("QoR Resource Summary (LUT/FF/DSP + BRAM)")
    ax_main.legend()

    ax_bram.bar(list(x), bram, width=0.55, color="tab:green", label="BRAM")
    ax_bram.set_ylabel("BRAM")
    ax_bram.set_xlabel("Top Module")
    ax_bram.set_xticks(list(x))
    ax_bram.set_xticklabels(tops, rotation=15)
    ax_bram.grid(axis="y", alpha=0.25)
    out = figures_dir / "qor_resources.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _make_onnx_mae_plot(suite: dict[str, str], figures_dir: Path) -> Path:
    labels = ["Q", "K", "V"]
    values = [
        _to_float(suite, "onnx_mae_q"),
        _to_float(suite, "onnx_mae_k"),
        _to_float(suite, "onnx_mae_v"),
    ]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, values)
    plt.ylabel("MAE")
    plt.title("ONNX vs Quantized Path MAE")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    out = figures_dir / "onnx_mae.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _make_dse_top5_plot(rows: list[dict[str, str]], figures_dir: Path) -> Path | None:
    if not rows:
        return None
    sorted_rows = sorted(rows, key=lambda r: float(r.get("score_tps_per_area", 0.0)), reverse=True)[:5]
    labels = [
        f"k{r.get('cfg_k_tile','?')}/pe{r.get('pe_mac_per_cycle','?')}/o{r.get('token_overhead_cycles','?')}"
        for r in sorted_rows
    ]
    values = [float(r.get("cycles_per_token", 0.0)) for r in sorted_rows]
    if not values:
        return None

    plt.figure(figsize=(9.0, 4.6))
    bars = plt.bar(labels, values)
    plt.ylabel("Cycles/token")
    plt.title("DSE Top-5 (by TPS/Area Score)")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    out = figures_dir / "dse_top5_cycles.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _make_dse_pareto_plot(rows: list[dict[str, str]], figures_dir: Path) -> Path | None:
    out = figures_dir / "dse_pareto.png"
    if not rows:
        plt.figure(figsize=(7.2, 4.8))
        plt.title("DSE Pareto Frontier")
        plt.text(0.5, 0.5, "No DSE data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    pareto_rows = [
        r for r in rows
        if str(r.get("pareto", "")).strip().lower() in {"1", "true", "yes"}
    ]
    if not pareto_rows:
        pareto_rows = rows
    try:
        area = [float(r.get("area_proxy", 0.0)) for r in pareto_rows]
        tps = [float(r.get("tps_estimate", 0.0)) for r in pareto_rows]
    except ValueError:
        return None
    if not area or not tps:
        return None

    order = np.argsort(np.array(area))
    area = [area[i] for i in order]
    tps = [tps[i] for i in order]

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(area, tps, s=45, alpha=0.9, label="Pareto candidates")
    if len(area) >= 2:
        plt.plot(area, tps, linewidth=1.2, alpha=0.8)
    plt.xlabel("Area proxy")
    plt.ylabel("TPS estimate")
    plt.title("DSE Pareto Frontier")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _make_calibration_plot(rows: list[dict[str, str]], figures_dir: Path) -> Path | None:
    if not rows:
        return None
    k_tiles = [int(float(r.get("cfg_k_tile", 0))) for r in rows]
    obs = [float(r.get("observed_cycles_per_token", 0.0)) for r in rows]
    raw = [float(r.get("predicted_cycles_per_token_raw", 0.0)) for r in rows]
    cal = [float(r.get("predicted_cycles_per_token_calibrated", 0.0)) for r in rows]
    if not k_tiles:
        return None

    order = np.argsort(np.array(k_tiles))
    k_tiles = [k_tiles[i] for i in order]
    obs = [obs[i] for i in order]
    raw = [raw[i] for i in order]
    cal = [cal[i] for i in order]

    plt.figure(figsize=(8.0, 4.6))
    plt.plot(k_tiles, obs, marker="o", label="Observed RTL")
    plt.plot(k_tiles, raw, marker="x", label="Model Raw")
    plt.plot(k_tiles, cal, marker="s", label="Model Calibrated")
    plt.xlabel("cfg_k_tile")
    plt.ylabel("Cycles/token")
    plt.title("Cycle Model Calibration")
    plt.grid(alpha=0.25)
    plt.legend()
    out = figures_dir / "cycle_calibration.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def _derive_metrics(
    suite: dict[str, str],
    qor_rows: list[dict[str, str]],
    rtl_flow: dict[str, object],
    dse_best: dict[str, object],
    calib: dict[str, object],
) -> dict[str, float]:
    tiny_cpu_tps = _to_float(suite, "tiny_cpu_tps")
    fpga_est_tps = _to_float(suite, "fpga_est_tps")
    scaleup_proxy_tps = _to_float(suite, "scaleup_proxy_tps")
    onnx_mae_q = _to_float(suite, "onnx_mae_q")
    onnx_mae_k = _to_float(suite, "onnx_mae_k")
    onnx_mae_v = _to_float(suite, "onnx_mae_v")
    speedup_fpga_est_vs_scaleup_proxy = (
        _to_float(suite, "speedup_fpga_est_vs_scaleup_proxy")
        if "speedup_fpga_est_vs_scaleup_proxy" in suite
        else (fpga_est_tps / scaleup_proxy_tps if scaleup_proxy_tps > 0 else 0.0)
    )
    speedup_fpga_est_vs_tiny_cpu = (
        _to_float(suite, "speedup_fpga_est_vs_tiny_cpu")
        if "speedup_fpga_est_vs_tiny_cpu" in suite
        else (fpga_est_tps / tiny_cpu_tps if tiny_cpu_tps > 0 else 0.0)
    )

    wns_values = [float(r["wns_ns"]) for r in qor_rows if (r.get("wns_ns") or "").strip()]
    max_lut = max((float(r["lut"] or 0.0) for r in qor_rows), default=0.0)
    max_ff = max((float(r["ff"] or 0.0) for r in qor_rows), default=0.0)
    max_dsp = max((float(r["dsp"] or 0.0) for r in qor_rows), default=0.0)

    metrics: dict[str, float] = {
        "tiny_cpu_tps": tiny_cpu_tps,
        "fpga_est_tps": fpga_est_tps,
        "scaleup_proxy_tps": scaleup_proxy_tps,
        "speedup_fpga_est_vs_scaleup_proxy": speedup_fpga_est_vs_scaleup_proxy,
        "speedup_fpga_est_vs_tiny_cpu": speedup_fpga_est_vs_tiny_cpu,
        "tiny_cpu_ms_per_token": (1000.0 / tiny_cpu_tps) if tiny_cpu_tps > 0 else 0.0,
        "fpga_est_ms_per_token": (1000.0 / fpga_est_tps) if fpga_est_tps > 0 else 0.0,
        "scaleup_proxy_ms_per_token": (1000.0 / scaleup_proxy_tps) if scaleup_proxy_tps > 0 else 0.0,
        "onnx_mae_q": onnx_mae_q,
        "onnx_mae_k": onnx_mae_k,
        "onnx_mae_v": onnx_mae_v,
        "onnx_mae_avg": mean([onnx_mae_q, onnx_mae_k, onnx_mae_v]),
        "qor_best_wns_ns": max(wns_values) if wns_values else 0.0,
        "qor_worst_wns_ns": min(wns_values) if wns_values else 0.0,
        "qor_max_lut": max_lut,
        "qor_max_ff": max_ff,
        "qor_max_dsp": max_dsp,
    }
    if rtl_flow:
        status = rtl_flow.get("status", {})
        if isinstance(status, dict):
            metrics["rtl_backend_perf_cycles"] = float(status.get("perf_cycles", 0.0))
            metrics["rtl_backend_perf_tokens"] = float(status.get("perf_tokens", 0.0))
            perf_tokens = float(status.get("perf_tokens", 0.0))
            perf_cycles = float(status.get("perf_cycles", 0.0))
            metrics["rtl_backend_cycles_per_token"] = (perf_cycles / perf_tokens) if perf_tokens > 0 else 0.0
    if dse_best:
        best = dse_best.get("best", {})
        if isinstance(best, dict):
            metrics["dse_best_cfg_k_tile"] = float(best.get("cfg_k_tile", 0.0))
            metrics["dse_best_pe_mac_per_cycle"] = float(best.get("pe_mac_per_cycle", 0.0))
            metrics["dse_best_token_overhead_cycles"] = float(best.get("token_overhead_cycles", 0.0))
            metrics["dse_best_cycles_per_token"] = float(best.get("cycles_per_token", 0.0))
            metrics["dse_best_score_tps_per_area"] = float(best.get("score_tps_per_area", 0.0))
        metrics["dse_trials"] = float(dse_best.get("num_trials", 0.0))
        metrics["dse_pareto_count"] = float(dse_best.get("pareto_count", 0.0))
    if calib:
        metrics["calib_scale"] = float(calib.get("scale", 1.0))
        metrics["calib_bias"] = float(calib.get("bias", 0.0))
        metrics["calib_mae_raw"] = float(calib.get("mae_raw", 0.0))
        metrics["calib_mae_calibrated"] = float(calib.get("mae_calibrated", 0.0))
        metrics["calib_improvement_pct"] = float(calib.get("improvement_pct", 0.0))
    return metrics


def _write_final_report(
    suite: dict[str, str],
    qor_rows: list[dict[str, str]],
    metrics: dict[str, float],
    out_dir: Path,
    opt_round_label: str,
    dse_best: dict[str, object],
    calib: dict[str, object],
    dse_fig: Path | None,
    dse_pareto_fig: Path | None,
    calib_fig: Path | None,
) -> Path:
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = out_dir / "final_report.md"

    qor_lines = [
        "| top | lut | ff | dsp | bram | uram | wns_ns |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in qor_rows:
        qor_lines.append(
            f"| {row['top']} | {row['lut']} | {row['ff']} | {row['dsp']} | {row['bram']} | {row['uram']} | {row['wns_ns']} |"
        )

    suite_timestamp = suite.get("timestamp_utc", "n/a")
    dse_best_row = dse_best.get("best", {}) if isinstance(dse_best, dict) else {}
    dse_lines = []
    if isinstance(dse_best_row, dict) and dse_best_row:
        dse_lines = [
            "",
            "## N8 DSE Summary",
            "",
            f"- trials: {int(float(metrics.get('dse_trials', 0.0)))}",
            f"- pareto_points: {int(float(metrics.get('dse_pareto_count', 0.0)))}",
            f"- best cfg_k_tile: {int(float(metrics.get('dse_best_cfg_k_tile', 0.0)))}",
            f"- best pe_mac_per_cycle: {int(float(metrics.get('dse_best_pe_mac_per_cycle', 0.0)))}",
            f"- best token_overhead_cycles: {int(float(metrics.get('dse_best_token_overhead_cycles', 0.0)))}",
            f"- best cycles_per_token: {metrics.get('dse_best_cycles_per_token', 0.0):.6f}",
            f"- best score_tps_per_area: {metrics.get('dse_best_score_tps_per_area', 0.0):.6f}",
        ]
        if dse_fig is not None:
            dse_lines += ["", "![DSE Top5](figures/dse_top5_cycles.png)"]
        if dse_pareto_fig is not None:
            dse_lines += ["", "![DSE Pareto](figures/dse_pareto.png)"]

    calib_lines = []
    if calib:
        calib_lines = [
            "",
            "## N9 Cycle Model Calibration",
            "",
            f"- scale: {metrics.get('calib_scale', 1.0):.6f}",
            f"- bias: {metrics.get('calib_bias', 0.0):.6f}",
            f"- mae_raw: {metrics.get('calib_mae_raw', 0.0):.6f}",
            f"- mae_calibrated: {metrics.get('calib_mae_calibrated', 0.0):.6f}",
            f"- improvement_pct: {metrics.get('calib_improvement_pct', 0.0):.2f}",
        ]
        if calib_fig is not None:
            calib_lines += ["", "![Calibration](figures/cycle_calibration.png)"]
    content = "\n".join(
        [
            "# Final Portfolio Report",
            "",
            f"- generated_utc: {generated_utc}",
            f"- benchmark_suite_timestamp_utc: {suite_timestamp}",
            f"- scope: Boardless LLM inference accelerator MVP + optimization round ({opt_round_label})",
            "",
            "## KPI Summary",
            "",
            "| KPI | Value |",
            "|---|---:|",
            f"| tiny_cpu_tps | {metrics['tiny_cpu_tps']:.6f} |",
            f"| fpga_est_tps | {metrics['fpga_est_tps']:.6f} |",
            f"| scaleup_proxy_tps | {metrics['scaleup_proxy_tps']:.6f} |",
            f"| speedup_fpga_est_vs_scaleup_proxy (primary) | {metrics['speedup_fpga_est_vs_scaleup_proxy']:.6f} |",
            f"| speedup_fpga_est_vs_tiny_cpu | {metrics['speedup_fpga_est_vs_tiny_cpu']:.6f} |",
            f"| tiny_cpu_ms_per_token | {metrics['tiny_cpu_ms_per_token']:.6f} |",
            f"| fpga_est_ms_per_token | {metrics['fpga_est_ms_per_token']:.6f} |",
            f"| onnx_mae_avg | {metrics['onnx_mae_avg']:.6f} |",
            f"| qor_best_wns_ns | {metrics['qor_best_wns_ns']:.6f} |",
            f"| rtl_backend_cycles_per_token | {metrics.get('rtl_backend_cycles_per_token', 0.0):.6f} |",
            "",
            "## Figures",
            "",
            "### Throughput (Primary KPI)",
            "![Throughput](figures/performance_tps.png)",
            "",
            "### Throughput (Reference, includes Tiny CPU)",
            "![Throughput All](figures/performance_all_tps.png)",
            "",
            "### QoR Resources",
            "![QoR](figures/qor_resources.png)",
            "",
            "### ONNX MAE",
            "![ONNX MAE](figures/onnx_mae.png)",
            "",
            "## QoR Table",
            "",
            *qor_lines,
            "",
            "## Validation Policy",
            "",
            "1. Run each validation step up to 10 times.",
            "2. Stop early when PASS is reached once.",
            "3. Record PASS/FAIL/BLOCKED in logs.",
            "",
            "## Scope Note",
            "",
            "- Current RTL is a boardless proxy-kernel implementation for pre-silicon bring-up.",
            "- Core pipeline and verification automation are real; full Transformer operator completeness is a next-phase target.",
            "- Primary speedup KPI uses fpga_est_tps vs scaleup_proxy_tps (same proxy scale).",
            *dse_lines,
            *calib_lines,
            "",
            "## Reproduce",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1",
            "```",
        ]
    )
    report.write_text(content + "\n", encoding="utf-8")
    return report


def _write_portfolio_runbook(out_dir: Path) -> Path:
    runbook = out_dir / "runbook.md"
    content = "\n".join(
        [
            "# Portfolio Runbook",
            "",
            "## Full Reproduction",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1",
            "```",
            "",
            "## Fast Path (N6 only)",
            "",
            "```powershell",
            "python scripts/generate_portfolio_assets.py",
            "python -m pytest tests/unit/test_portfolio_packaging.py -q",
            "```",
            "",
            "## Commit-ready Artifacts",
            "",
            "- `README.md`",
            "- `docs/portfolio/final_report.md`",
            "- `docs/portfolio/manifest.json`",
            "- `docs/portfolio/figures/*.png`",
            "",
            "## Logs",
            "",
            "- `results/step_validation_runs.csv`",
            "- `results/boardless_progress_log.csv`",
            "- `logs/boardless_execution_log.md`",
        ]
    )
    runbook.write_text(content + "\n", encoding="utf-8")
    return runbook


def _write_readme(metrics: dict[str, float], readme_path: Path, opt_round_label: str) -> Path:
    dse_line = ""
    if metrics.get("dse_trials", 0.0) > 0:
        dse_line = (
            f"- dse_best(k_tile/pe/overhead): "
            f"{int(metrics.get('dse_best_cfg_k_tile', 0.0))}/"
            f"{int(metrics.get('dse_best_pe_mac_per_cycle', 0.0))}/"
            f"{int(metrics.get('dse_best_token_overhead_cycles', 0.0))}"
        )
    calib_line = ""
    if "calib_improvement_pct" in metrics:
        calib_line = f"- cycle_model_calibration_improvement_pct: {metrics.get('calib_improvement_pct', 0.0):.2f}"

    content = "\n".join(
        [x for x in [
            "# Transformer Acceleration (Boardless LLM Inference)",
            "",
            "Repository for boardless development, validation, and portfolio packaging",
            "of an FPGA/NPU-style LLM inference accelerator.",
            "",
            "## Current Status",
            "",
            "- Boardless track: B1~B8 PASS",
            f"- Optimization round: {opt_round_label} PASS",
            f"- tiny_cpu_tps: {metrics['tiny_cpu_tps']:.6f}",
            f"- fpga_est_tps: {metrics['fpga_est_tps']:.6f}",
            f"- scaleup_proxy_tps: {metrics['scaleup_proxy_tps']:.6f}",
            f"- speedup_fpga_est_vs_scaleup_proxy: {metrics['speedup_fpga_est_vs_scaleup_proxy']:.6f} (primary)",
            f"- onnx_mae_avg: {metrics['onnx_mae_avg']:.6f}",
            dse_line,
            calib_line,
            "",
            "## Quick Start",
            "",
            "```powershell",
            "python -m pip install -r requirements-boardless.txt",
            "powershell -ExecutionPolicy Bypass -File scripts/run_validation_10x.ps1",
            "python -m pytest tests -q",
            "```",
            "",
            "## Reproduce Portfolio Package",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1",
            "```",
            "",
            "## P1 RTL Backend Path",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File scripts/run_n7_rtl_backend.ps1 -MaxRuns 10",
            "python scripts/run_rtl_backend_flow.py",
            "```",
            "",
            "## P2 DSE/Autotune",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File scripts/run_n8_dse.ps1 -MaxRuns 10",
            "python scripts/run_dse_autotune.py",
            "```",
            "",
            "## N9 Cycle-model Calibration",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File scripts/run_n9_calibration.ps1 -MaxRuns 10",
            "python scripts/calibrate_cycle_model.py",
            "```",
            "",
            "## Outputs",
            "",
            "- Final report: `docs/portfolio/final_report.md`",
            "- Figures: `docs/portfolio/figures/`",
            "- Manifest: `docs/portfolio/manifest.json`",
            "- Runbook: `docs/portfolio/runbook.md`",
            "",
            "## Notes",
            "",
            "- `results/` and `logs/` are ignored by default and regenerated per run.",
            "- Commit-facing portfolio artifacts are under `docs/portfolio/`.",
            "",
            "## Implementation Scope (Current)",
            "",
            "- This repository currently uses proxy RTL kernels for boardless pre-silicon bring-up.",
            "- Runtime default path is NumPy backend; RTL path is validated via cocotb/unit tests.",
            "- Main KPI for cross-scale fairness is `fpga_est_tps / scaleup_proxy_tps`.",
            "",
            "## Visualization Results",
            "",
            "### Throughput (Primary KPI)",
            "![Throughput](docs/portfolio/figures/performance_tps.png)",
            "",
            "### Throughput (Reference, includes Tiny CPU)",
            "![Throughput All](docs/portfolio/figures/performance_all_tps.png)",
            "",
            "### QoR Resources",
            "![QoR](docs/portfolio/figures/qor_resources.png)",
            "",
            "### ONNX MAE",
            "![ONNX MAE](docs/portfolio/figures/onnx_mae.png)",
            "",
            "### DSE Top-5 (if available)",
            "![DSE Top5](docs/portfolio/figures/dse_top5_cycles.png)",
            "",
            "### DSE Pareto (if available)",
            "![DSE Pareto](docs/portfolio/figures/dse_pareto.png)",
            "",
            "### Cycle Model Calibration (if available)",
            "![Calibration](docs/portfolio/figures/cycle_calibration.png)",
        ] if x != ""]
    )
    readme_path.write_text(content + "\n", encoding="utf-8")
    return readme_path


def _build_manifest(
    report: Path,
    readme: Path,
    runbook: Path,
    figures: list[Path],
    suite_csv: Path,
    qor_csv: Path,
    metrics: dict[str, float],
) -> dict[str, object]:
    artifacts = [report, readme, runbook, *figures]
    artifact_entries = [
        {"path": _as_rel(p), "sha256": _sha256(p), "size_bytes": p.stat().st_size}
        for p in artifacts
    ]

    manifest: dict[str, object] = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "benchmark_suite_csv": _as_rel(suite_csv),
            "qor_summary_csv": _as_rel(qor_csv),
        },
        "metrics": metrics,
        "artifacts": artifact_entries,
        "commands": {
            "reproduce_full": "powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1",
            "reproduce_n6_only": "python scripts/generate_portfolio_assets.py",
        },
        # Backward-compatible keys used by existing tooling/tests.
        "final_report": _as_rel(report),
        "readme": _as_rel(readme),
        "figures": [_as_rel(p) for p in figures],
    }
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio README/report/figures/manifest.")
    parser.add_argument(
        "--suite-csv",
        default=str(DEFAULT_RESULTS / "benchmark_suite.csv"),
        help="Path to benchmark suite csv (item,value format).",
    )
    parser.add_argument(
        "--qor-csv",
        default=str(DEFAULT_RESULTS / "qor_summary.csv"),
        help="Path to synthesis QoR summary csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where portfolio outputs are written.",
    )
    parser.add_argument(
        "--readme-path",
        default=str(ROOT / "README.md"),
        help="Path to root README output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    suite_csv = Path(args.suite_csv).resolve()
    qor_csv = Path(args.qor_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    readme_path = Path(args.readme_path).resolve()
    figures_dir = out_dir / "figures"

    suite = _read_kv_csv(suite_csv)
    _validate_suite(suite)
    qor_rows = _read_qor(qor_csv)
    rtl_flow = _read_optional_json(DEFAULT_RESULTS / "rtl_backend_flow_result.json")
    dse_best = _read_optional_json(DEFAULT_RESULTS / "dse_autotune_best.json")
    dse_rows = _read_optional_csv_rows(DEFAULT_RESULTS / "dse_autotune.csv")
    dse_pareto_rows = _read_optional_csv_rows(DEFAULT_RESULTS / "dse_pareto.csv")
    calib = _read_optional_json(DEFAULT_RESULTS / "model_calibration.json")
    calib_rows = _read_optional_csv_rows(DEFAULT_RESULTS / "model_calibration.csv")
    opt_round_label = _detect_optimization_round(DEFAULT_RESULTS / "boardless_progress_log.csv")
    metrics = _derive_metrics(suite, qor_rows, rtl_flow, dse_best, calib)

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    perf_fig = _make_perf_plot(suite, figures_dir)
    perf_all_fig = _make_perf_all_plot(suite, figures_dir)
    qor_fig = _make_qor_plot(qor_rows, figures_dir)
    mae_fig = _make_onnx_mae_plot(suite, figures_dir)
    dse_fig = _make_dse_top5_plot(dse_rows, figures_dir)
    dse_pareto_fig = _make_dse_pareto_plot(dse_pareto_rows, figures_dir)
    calib_fig = _make_calibration_plot(calib_rows, figures_dir)
    figures = [perf_fig, perf_all_fig, qor_fig, mae_fig]
    if dse_fig is not None:
        figures.append(dse_fig)
    if dse_pareto_fig is not None:
        figures.append(dse_pareto_fig)
    if calib_fig is not None:
        figures.append(calib_fig)

    report = _write_final_report(
        suite,
        qor_rows,
        metrics,
        out_dir,
        opt_round_label=opt_round_label,
        dse_best=dse_best,
        calib=calib,
        dse_fig=dse_fig,
        dse_pareto_fig=dse_pareto_fig,
        calib_fig=calib_fig,
    )
    runbook = _write_portfolio_runbook(out_dir)
    readme = _write_readme(metrics, readme_path, opt_round_label=opt_round_label)

    manifest = _build_manifest(
        report=report,
        readme=readme,
        runbook=runbook,
        figures=figures,
        suite_csv=suite_csv,
        qor_csv=qor_csv,
        metrics=metrics,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("portfolio assets generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
