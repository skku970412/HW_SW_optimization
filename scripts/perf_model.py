#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class PerfInput:
    layers: int
    hidden: int
    seq: int
    pe_mac_per_cycle: int
    clock_mhz: float
    efficiency: float


@dataclass
class PerfOutput:
    mac_per_token: float
    ideal_tokens_per_sec: float
    effective_tokens_per_sec: float
    cycles_per_token_effective: float


def estimate(inp: PerfInput) -> PerfOutput:
    # Approximation used in docs/spec.md.
    mac_per_token = inp.layers * ((12 * (inp.hidden**2)) + (2 * inp.hidden * inp.seq))
    peak_mac_per_sec = inp.pe_mac_per_cycle * inp.clock_mhz * 1e6
    ideal_tokens_per_sec = peak_mac_per_sec / mac_per_token
    effective_tokens_per_sec = ideal_tokens_per_sec * inp.efficiency
    cycles_per_token_effective = (inp.clock_mhz * 1e6) / effective_tokens_per_sec
    return PerfOutput(
        mac_per_token=mac_per_token,
        ideal_tokens_per_sec=ideal_tokens_per_sec,
        effective_tokens_per_sec=effective_tokens_per_sec,
        cycles_per_token_effective=cycles_per_token_effective,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Boardless performance model estimator.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--pe-mac-per-cycle", type=int, default=256)
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    parser.add_argument("--efficiency", type=float, default=0.15)
    args = parser.parse_args()

    inp = PerfInput(
        layers=args.layers,
        hidden=args.hidden,
        seq=args.seq,
        pe_mac_per_cycle=args.pe_mac_per_cycle,
        clock_mhz=args.clock_mhz,
        efficiency=args.efficiency,
    )
    out = estimate(inp)

    print("mac_per_token=", int(out.mac_per_token))
    print("ideal_tokens_per_sec=", round(out.ideal_tokens_per_sec, 3))
    print("effective_tokens_per_sec=", round(out.effective_tokens_per_sec, 3))
    print("cycles_per_token_effective=", round(out.cycles_per_token_effective, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
