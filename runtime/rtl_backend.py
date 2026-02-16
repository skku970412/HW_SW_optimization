from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from runtime import register_map as rm
from runtime.np_kernels import KVCache, attention_decode_step, gemm_int8w_int16a_acc32, requantize_int16


def _pack_error_code(text: str) -> int:
    # Stable small error signature for REG_LAST_ERROR.
    b = text.encode("utf-8", errors="ignore")
    return int(sum(b) & 0xFFFFFFFF)


class RtlBackend:
    """
    Boardless RTL backend proxy.
    - Exposes MMIO-like register behavior.
    - Runs functional path with Python golden kernels.
    - Emits cycle/stall counters using a simple token-level cycle model.
    """

    def __init__(
        self,
        *,
        dim: int,
        max_seq: int,
        cfg_k_tile: int = 16,
        pe_mac_per_cycle: int = 256,
        token_overhead_cycles: int = 12,
        cycle_calib_scale: float = 1.0,
        cycle_calib_bias: float = 0.0,
    ) -> None:
        self.dim = dim
        self.max_seq = max_seq
        self.cfg_k_tile = max(1, int(cfg_k_tile))
        self.pe_mac_per_cycle = max(1, int(pe_mac_per_cycle))
        self.token_overhead_cycles = max(1, int(token_overhead_cycles))
        self.cycle_calib_scale = float(cycle_calib_scale)
        self.cycle_calib_bias = float(cycle_calib_bias)

        self.regs: dict[int, int] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.cache = KVCache(max_seq=max_seq, dim=dim)
        self.last_error = ""
        self.init()

    def init(self) -> None:
        self.regs = {
            rm.REG_CONTROL: 0,
            rm.REG_STATUS: 0,
            rm.REG_PROMPT_LEN: 0,
            rm.REG_GEN_LEN: 0,
            rm.REG_DONE_TOKENS: 0,
            rm.REG_LAST_ERROR: 0,
            rm.REG_PERF_CYCLES: 0,
            rm.REG_PERF_TOKENS: 0,
            rm.REG_PERF_STALL_IN: 0,
            rm.REG_PERF_STALL_OUT: 0,
            rm.REG_CFG_K_TILE: self.cfg_k_tile,
        }
        self.cache = KVCache(max_seq=self.max_seq, dim=self.dim)
        self.last_error = ""

    def load(self, pack_dir: Path | str) -> None:
        p = Path(pack_dir)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        if int(meta["dim"]) != self.dim:
            raise ValueError("pack dim mismatch")
        self.weights["w_q"] = np.load(p / "w_q_int8.npy")
        self.weights["w_k"] = np.load(p / "w_k_int8.npy")
        self.weights["w_v"] = np.load(p / "w_v_int8.npy")
        self.weights["dequant_scale"] = np.array([meta["dequant_scale"]], dtype=np.float32)

    def mmio_write(self, addr: int, value: int) -> None:
        if addr == rm.REG_CONTROL:
            self.regs[rm.REG_CONTROL] = int(value) & 0xFFFFFFFF
            if value & rm.CTRL_RESET:
                self.init()
        elif addr in self.regs:
            self.regs[addr] = int(value) & 0xFFFFFFFF

    def mmio_read(self, addr: int) -> int:
        return int(self.regs.get(addr, 0))

    def _estimate_token_cycles(self, seq_len: int) -> tuple[int, int, int]:
        k_tile = max(1, int(self.regs.get(rm.REG_CFG_K_TILE, self.cfg_k_tile)))
        k_pass = int(np.ceil(self.dim / float(k_tile)))

        # GEMM(q,k,v): 3 * D*D scaled by K-tiling pass count.
        gemm_macs = 3 * self.dim * self.dim * k_pass
        attn_macs = 2 * seq_len * self.dim
        mac_cycles = int(np.ceil((gemm_macs + attn_macs) / float(self.pe_mac_per_cycle)))
        stall_in = max(0, seq_len // 32) + max(0, (8 - min(k_tile, 8)))
        stall_out = 1 if (seq_len % 64 == 0 and seq_len > 0) else 0
        raw_total = self.token_overhead_cycles + mac_cycles + stall_in + stall_out
        calibrated = int(round(raw_total * self.cycle_calib_scale + self.cycle_calib_bias))
        total = max(1, calibrated)
        return total, stall_in, stall_out

    def run(self, prompt_tokens: np.ndarray, gen_len: int) -> np.ndarray:
        try:
            self.mmio_write(rm.REG_CONTROL, rm.CTRL_START)
            self.regs[rm.REG_STATUS] = rm.STATUS_BUSY
            self.regs[rm.REG_PROMPT_LEN] = int(prompt_tokens.shape[0])
            self.regs[rm.REG_GEN_LEN] = int(gen_len)
            self.regs[rm.REG_DONE_TOKENS] = 0
            self.regs[rm.REG_PERF_CYCLES] = 0
            self.regs[rm.REG_PERF_TOKENS] = 0
            self.regs[rm.REG_PERF_STALL_IN] = 0
            self.regs[rm.REG_PERF_STALL_OUT] = 0
            self.regs[rm.REG_LAST_ERROR] = 0

            if prompt_tokens.ndim != 2 or prompt_tokens.shape[1] != self.dim:
                raise ValueError("prompt shape must be [T, D]")
            if gen_len <= 0:
                raise ValueError("gen_len must be > 0")

            x_t = prompt_tokens[-1].astype(np.int16)
            scale = float(self.weights["dequant_scale"][0])
            outputs: list[np.ndarray] = []

            for _ in range(gen_len):
                q = gemm_int8w_int16a_acc32(x_t.reshape(1, -1), self.weights["w_q"]).reshape(-1).astype(np.float32)
                k = gemm_int8w_int16a_acc32(x_t.reshape(1, -1), self.weights["w_k"]).reshape(-1).astype(np.float32)
                v = gemm_int8w_int16a_acc32(x_t.reshape(1, -1), self.weights["w_v"]).reshape(-1).astype(np.float32)

                self.cache.append(k, v)
                k_all, v_all = self.cache.get()
                y = attention_decode_step(q, k_all, v_all)
                y_int16 = requantize_int16(np.round(y).astype(np.int32), scale=scale)

                seq_len = int(self.cache.length)
                cycles, stall_in, stall_out = self._estimate_token_cycles(seq_len)
                self.regs[rm.REG_PERF_CYCLES] += cycles
                self.regs[rm.REG_PERF_TOKENS] += 1
                self.regs[rm.REG_PERF_STALL_IN] += stall_in
                self.regs[rm.REG_PERF_STALL_OUT] += stall_out
                self.regs[rm.REG_DONE_TOKENS] += 1

                outputs.append(y_int16.copy())
                x_t = y_int16

            self.regs[rm.REG_STATUS] = rm.STATUS_DONE
            return np.stack(outputs, axis=0)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.regs[rm.REG_LAST_ERROR] = _pack_error_code(self.last_error)
            self.regs[rm.REG_STATUS] = rm.STATUS_ERROR
            raise

    def poll(self) -> dict[str, int | str]:
        return {
            "backend": "rtl_proxy",
            "status": self.regs.get(rm.REG_STATUS, 0),
            "done_tokens": self.regs.get(rm.REG_DONE_TOKENS, 0),
            "prompt_len": self.regs.get(rm.REG_PROMPT_LEN, 0),
            "gen_len": self.regs.get(rm.REG_GEN_LEN, 0),
            "perf_cycles": self.regs.get(rm.REG_PERF_CYCLES, 0),
            "perf_tokens": self.regs.get(rm.REG_PERF_TOKENS, 0),
            "perf_stall_in": self.regs.get(rm.REG_PERF_STALL_IN, 0),
            "perf_stall_out": self.regs.get(rm.REG_PERF_STALL_OUT, 0),
            "last_error_code": self.regs.get(rm.REG_LAST_ERROR, 0),
        }
