from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "hw" / "rtl"


def _find_iverilog() -> str | None:
    exe = shutil.which("iverilog")
    if exe:
        return exe
    fallback = Path(r"C:\iverilog\bin\iverilog.exe")
    return str(fallback) if fallback.exists() else None


@pytest.mark.parametrize(
    "top",
    ["gemm_core", "attention_core", "kv_cache", "decoder_block_top", "npu_top"],
)
def test_rtl_compiles_with_iverilog(top: str):
    iverilog = _find_iverilog()
    if not iverilog:
        pytest.skip("iverilog not found")

    out = ROOT / "tests" / "tb" / f"{top}.out"
    cmd = [
        iverilog,
        "-g2012",
        "-s",
        top,
        "-o",
        str(out),
        str(RTL / "gemm_core.sv"),
        str(RTL / "attention_core.sv"),
        str(RTL / "kv_cache.sv"),
        str(RTL / "decoder_block_top.sv"),
        str(RTL / "npu_top.sv"),
    ]
    subprocess.run(cmd, check=True)
    assert out.exists()
