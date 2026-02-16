#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results/boardless_progress_log.csv"
OUT_PATH = ROOT / "results/boardless_status.md"


def main() -> int:
    if not CSV_PATH.exists():
        OUT_PATH.write_text(
            "# Boardless Status\n\nNo progress log found.\n", encoding="utf-8"
        )
        print("summary generated (empty)")
        return 0

    latest_by_step: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    with CSV_PATH.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = row["step"]
            latest_by_step[key] = row

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Boardless Status",
        "",
        f"- generated_utc: {now}",
        "",
        "## Latest Step Status",
        "",
        "| week | step | status | validation_runs | summary | blocker |",
        "|---|---|---|---:|---|---|",
    ]

    for row in latest_by_step.values():
        lines.append(
            f"| {row['week']} | {row['step']} | {row['status']} | {row['validation_runs']} | "
            f"{row['summary']} | {row['blocker']} |"
        )

    blocked = [r for r in latest_by_step.values() if r["status"] == "BLOCKED"]
    lines += ["", "## Blockers", ""]
    if not blocked:
        lines.append("- none")
    else:
        for r in blocked:
            lines.append(f"- {r['week']} / {r['step']}: {r['blocker']}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
