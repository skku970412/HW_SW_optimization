from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner


ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / "hw" / "rtl"


def _prepare_iverilog_path() -> bool:
    fallback_dir = Path(r"C:\iverilog\bin")
    if shutil.which("iverilog"):
        return True
    if fallback_dir.exists():
        os.environ["PATH"] = str(fallback_dir) + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("iverilog") is not None
    return False


def _run_cocotb(top: str, module: str) -> None:
    runner = get_runner("icarus")
    build_dir = ROOT / "sim_build" / top
    verilog_sources = [
        RTL_DIR / "gemm_core.sv",
        RTL_DIR / "attention_core.sv",
        RTL_DIR / "kv_cache.sv",
        RTL_DIR / "decoder_block_top.sv",
    ]
    runner.build(
        sources=verilog_sources,
        hdl_toplevel=top,
        build_dir=str(build_dir),
        always=True,
    )
    runner.test(
        test_module=module,
        hdl_toplevel=top,
        build_dir=str(build_dir),
        test_dir=str(build_dir / "test"),
    )


@pytest.mark.skipif(not _prepare_iverilog_path(), reason="iverilog not found")
def test_cocotb_gemm():
    _run_cocotb(top="gemm_core", module="tests.tb.cocotb_gemm")


@pytest.mark.skipif(not _prepare_iverilog_path(), reason="iverilog not found")
def test_cocotb_attention():
    _run_cocotb(top="attention_core", module="tests.tb.cocotb_attention")


@pytest.mark.skipif(not _prepare_iverilog_path(), reason="iverilog not found")
def test_cocotb_kvcache():
    _run_cocotb(top="kv_cache", module="tests.tb.cocotb_kvcache")
