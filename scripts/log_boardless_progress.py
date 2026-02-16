#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_LOG = ROOT / "results/boardless_progress_log.csv"
MD_LOG = ROOT / "logs/boardless_execution_log.md"


def ensure_logs() -> None:
    CSV_LOG.parent.mkdir(parents=True, exist_ok=True)
    MD_LOG.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_LOG.exists():
        with CSV_LOG.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "timestamp_utc",
                    "week",
                    "step",
                    "status",
                    "validation_runs",
                    "summary",
                    "blocker",
                ]
            )

    if not MD_LOG.exists():
        MD_LOG.write_text("# Boardless Execution Log\n", encoding="utf-8")


def append_csv(
    timestamp_utc: str,
    week: str,
    step: str,
    status: str,
    validation_runs: int,
    summary: str,
    blocker: str,
) -> None:
    with CSV_LOG.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [timestamp_utc, week, step, status, validation_runs, summary, blocker]
        )


def append_md(
    timestamp_utc: str,
    week: str,
    step: str,
    status: str,
    validation_runs: int,
    summary: str,
    blocker: str,
) -> None:
    lines = [
        f"\n## {timestamp_utc} {week} - {step}",
        f"- status: {status}",
        f"- validation_runs: {validation_runs}",
        f"- summary: {summary}",
    ]
    if blocker:
        lines.append(f"- blocker: {blocker}")
    with MD_LOG.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append boardless progress logs.")
    parser.add_argument("--week", required=True, help="Week label, e.g., B1")
    parser.add_argument("--step", required=True, help="Step name")
    parser.add_argument("--status", required=True, help="PASS/FAIL/BLOCKED/INFO")
    parser.add_argument(
        "--validation-runs",
        type=int,
        default=0,
        help="Number of validation attempts used",
    )
    parser.add_argument("--summary", required=True, help="Short summary")
    parser.add_argument("--blocker", default="", help="Blocker details")
    args = parser.parse_args()

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ensure_logs()
    append_csv(
        timestamp_utc,
        args.week,
        args.step,
        args.status,
        args.validation_runs,
        args.summary,
        args.blocker,
    )
    append_md(
        timestamp_utc,
        args.week,
        args.step,
        args.status,
        args.validation_runs,
        args.summary,
        args.blocker,
    )
    print("logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
