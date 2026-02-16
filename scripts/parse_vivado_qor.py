#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_util(text: str, keys: list[str]) -> str:
    for key in keys:
        # Matches table rows like: | CLB LUTs* | 123 | ...
        pat = rf"^\|\s*{re.escape(key)}\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|"
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return m.group(1)
    return ""


def _extract_wns(text: str) -> str:
    # Timing summary headers often contain WNS(ns). Prefer explicit value line.
    m = re.search(r"WNS\(ns\)\s*:\s*([\-0-9.]+)", text)
    if m:
        return m.group(1)

    # Common Vivado table:
    #   WNS(ns) ... 
    #   ------- ...
    #    2.997  0.000 ...
    table = re.search(
        r"WNS\(ns\).*?\n\s*-+\s+-+.*?\n\s*([A-Za-z0-9\.\-]+)\s+([A-Za-z0-9\.\-]+)",
        text,
        flags=re.DOTALL,
    )
    if table:
        return table.group(1)

    # Fallback: line containing only numeric table entries after a WNS header.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and i + 1 < len(lines):
            for j in range(i + 1, min(i + 8, len(lines))):
                token_match = re.findall(r"(NA|[\-]?\d+\.\d+|[\-]?\d+)", lines[j])
                if token_match and "-------" not in lines[j]:
                    return token_match[0]
    return ""


def parse_top(top_dir: Path) -> dict[str, str]:
    util_rpt = top_dir / "utilization.rpt"
    timing_rpt = top_dir / "timing_summary.rpt"

    if not util_rpt.exists() or not timing_rpt.exists():
        raise FileNotFoundError(f"missing reports in {top_dir}")

    util_text = _read_text(util_rpt)
    timing_text = _read_text(timing_rpt)

    return {
        "top": top_dir.name,
        "lut": _extract_util(util_text, ["CLB LUTs*", "Slice LUTs*"]),
        "ff": _extract_util(util_text, ["CLB Registers", "Slice Registers"]),
        "dsp": _extract_util(util_text, ["DSPs"]),
        "bram": _extract_util(util_text, ["Block RAM Tile"]),
        "uram": _extract_util(util_text, ["URAM"]),
        "wns_ns": _extract_wns(timing_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Vivado QoR reports into CSV.")
    parser.add_argument("--qor-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--part", required=True)
    parser.add_argument("--clock-period-ns", required=True)
    args = parser.parse_args()

    tops = sorted([p for p in args.qor_dir.iterdir() if p.is_dir()])
    rows: list[dict[str, str]] = []
    for top_dir in tops:
        try:
            row = parse_top(top_dir)
        except FileNotFoundError:
            continue
        row["part"] = args.part
        row["target_clock_period_ns"] = str(args.clock_period_ns)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "top",
                "part",
                "target_clock_period_ns",
                "lut",
                "ff",
                "dsp",
                "bram",
                "uram",
                "wns_ns",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"parsed {len(rows)} tops into {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
