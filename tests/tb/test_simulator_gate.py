from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def test_rtl_simulator_available_for_cocotb():
    simulators = ["verilator", "iverilog", "xsim", "ghdl"]
    found = [s for s in simulators if shutil.which(s)]
    if Path(r"C:\iverilog\bin\iverilog.exe").exists() and "iverilog" not in found:
        found.append("iverilog")
    if not found:
        pytest.skip("No RTL simulator detected. Cocotb RTL runs are blocked in this environment.")
    assert len(found) >= 1
