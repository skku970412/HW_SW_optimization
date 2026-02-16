from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def gemm_core_dot_product(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.cfg_start.value = 0
    dut.in_valid.value = 0
    dut.out_ready.value = 1
    dut.a_data.value = 0
    dut.b_data.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    # Start K_TILE accumulation window.
    dut.cfg_start.value = 1
    await RisingEdge(dut.clk)
    dut.cfg_start.value = 0

    expected = 0
    for i in range(16):
        a = i + 1
        b = 2
        expected += a * b
        dut.in_valid.value = 1
        dut.a_data.value = a
        dut.b_data.value = b
        await RisingEdge(dut.clk)

    dut.in_valid.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) == 1:
            got = int(dut.out_data.value.signed_integer)
            assert got == expected, f"gemm out mismatch: got={got}, expected={expected}"
            return

    raise AssertionError("gemm_core out_valid did not assert in expected cycles")

