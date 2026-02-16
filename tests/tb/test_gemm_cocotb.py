from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def gemm_core_smoke(dut):
    """Smoke test for gemm_core skeleton."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    dut.cfg_start.value = 0
    dut.a_data.value = 0
    dut.b_data.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    # Start a short accumulation.
    dut.cfg_start.value = 1
    await RisingEdge(dut.clk)
    dut.cfg_start.value = 0

    dut.in_valid.value = 1
    dut.a_data.value = 3
    dut.b_data.value = 4
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    dut.out_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert int(dut.out_valid.value) in (0, 1)
