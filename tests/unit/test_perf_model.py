from __future__ import annotations

from scripts.perf_model import PerfInput, estimate


def test_perf_model_reference_case():
    out = estimate(
        PerfInput(
            layers=6,
            hidden=768,
            seq=256,
            pe_mac_per_cycle=256,
            clock_mhz=200.0,
            efficiency=0.15,
        )
    )
    assert int(out.mac_per_token) == 44826624
    assert 100.0 < out.effective_tokens_per_sec < 250.0
    assert out.cycles_per_token_effective > 0
