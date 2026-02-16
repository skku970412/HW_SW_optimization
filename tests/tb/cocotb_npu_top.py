from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def _mmio_write(dut, addr: int, data: int) -> None:
    dut.mmio_addr.value = addr
    dut.mmio_wdata.value = data
    dut.mmio_wr_en.value = 1
    dut.mmio_rd_en.value = 0
    await RisingEdge(dut.clk)
    dut.mmio_wr_en.value = 0
    dut.mmio_wdata.value = 0


async def _mmio_read(dut, addr: int) -> int:
    dut.mmio_addr.value = addr
    dut.mmio_rd_en.value = 1
    dut.mmio_wr_en.value = 0
    await RisingEdge(dut.clk)
    val = int(dut.mmio_rdata.value)
    dut.mmio_rd_en.value = 0
    return val


@cocotb.test()
async def npu_top_mmio_run(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    dut.mmio_wr_en.value = 0
    dut.mmio_rd_en.value = 0
    dut.mmio_addr.value = 0
    dut.mmio_wdata.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    # Program minimal job.
    await _mmio_write(dut, 0x08, 4)   # prompt_len
    await _mmio_write(dut, 0x0C, 3)   # gen_len
    await _mmio_write(dut, 0x28, 2)   # cfg_k_tile (forces stall_in in this proxy model)
    await _mmio_write(dut, 0x00, 0x1) # start

    for _ in range(256):
        status = await _mmio_read(dut, 0x04)
        if status & 0x2:
            break
    else:
        raise AssertionError("npu_top did not reach DONE status")

    done_tokens = await _mmio_read(dut, 0x10)
    perf_cycles = await _mmio_read(dut, 0x18)
    perf_stall_in = await _mmio_read(dut, 0x20)

    assert done_tokens == 3, f"done_tokens mismatch: {done_tokens}"
    assert perf_cycles > 0, "perf_cycles should be > 0"
    assert perf_stall_in > 0, "perf_stall_in should be > 0 when cfg_k_tile is low"
