from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def kv_cache_write_read_and_bypass(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.rd_en.value = 0
    dut.wr_addr.value = 0
    dut.rd_addr.value = 0
    dut.k_wr_data.value = 0
    dut.v_wr_data.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    # Write addr=3 then read back.
    dut.wr_en.value = 1
    dut.wr_addr.value = 3
    dut.k_wr_data.value = 11
    dut.v_wr_data.value = 22
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0

    dut.rd_en.value = 1
    dut.rd_addr.value = 3
    await RisingEdge(dut.clk)
    dut.rd_en.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.k_rd_data.value) == 11
    assert int(dut.v_rd_data.value) == 22

    # Same-cycle write/read collision should return write data (write-first).
    dut.wr_en.value = 1
    dut.rd_en.value = 1
    dut.wr_addr.value = 5
    dut.rd_addr.value = 5
    dut.k_wr_data.value = 33
    dut.v_wr_data.value = 44
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    dut.rd_en.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.k_rd_data.value) == 33
    assert int(dut.v_rd_data.value) == 44

