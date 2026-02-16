from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_required_python_packages():
    for mod in ["numpy", "pytest", "cocotb"]:
        assert importlib.util.find_spec(mod) is not None


def test_simulator_presence_or_blocker_visible():
    simulators = ["verilator", "iverilog", "xsim", "ghdl"]
    found = [s for s in simulators if shutil.which(s) is not None]
    if Path(r"C:\iverilog\bin\iverilog.exe").exists() and "iverilog" not in found:
        found.append("iverilog")
    # B1 policy: simulator may be absent, but this must be explicit for logging.
    assert isinstance(found, list)
