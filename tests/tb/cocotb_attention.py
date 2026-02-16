from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def attention_core_weighted_accum(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.cfg_start.value = 0
    dut.in_valid.value = 0
    dut.out_ready.value = 1
    dut.q_data.value = 0
    dut.k_data.value = 0
    dut.v_data.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    dut.cfg_start.value = 1
    await RisingEdge(dut.clk)
    dut.cfg_start.value = 0

    # q*k*v = 2*3*4 = 24, K_TILE=16 => value_acc=384, shift 8 => 1
    for _ in range(16):
        dut.in_valid.value = 1
        dut.q_data.value = 2
        dut.k_data.value = 3
        dut.v_data.value = 4
        await RisingEdge(dut.clk)

    dut.in_valid.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) == 1:
            got = int(dut.out_data.value.signed_integer)
            assert got == 1, f"attention out mismatch: got={got}, expected=1"
            return

    raise AssertionError("attention_core out_valid did not assert in expected cycles")

