from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime import register_map as rm
from runtime.np_kernels import KVCache, attention_decode_step, gemm_int8w_int16a_acc32, requantize_int16
from runtime.rtl_backend import RtlBackend


@dataclass
class RuntimeConfig:
    dim: int = 16
    max_seq: int = 256
    backend: str = "numpy"
    cfg_k_tile: int = 16
    pe_mac_per_cycle: int = 256
    token_overhead_cycles: int = 12
    cycle_calib_scale: float = 1.0
    cycle_calib_bias: float = 0.0


class BoardlessNpuRuntime:
    """
    Runtime API shape:
    - init()
    - load(pack_dir)
    - run(prompt_tokens, gen_len)
    - poll()
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.regs: dict[int, int] = {}
        self.weights: dict[str, np.ndarray] = {}
        self.cache = KVCache(max_seq=self.config.max_seq, dim=self.config.dim)
        self.generated: list[np.ndarray] = []
        self.last_error = ""
        self._rtl_backend: RtlBackend | None = None
        if self.config.backend == "rtl":
            self._rtl_backend = RtlBackend(
                dim=self.config.dim,
                max_seq=self.config.max_seq,
                cfg_k_tile=self.config.cfg_k_tile,
                pe_mac_per_cycle=self.config.pe_mac_per_cycle,
                token_overhead_cycles=self.config.token_overhead_cycles,
                cycle_calib_scale=self.config.cycle_calib_scale,
                cycle_calib_bias=self.config.cycle_calib_bias,
            )
        elif self.config.backend != "numpy":
            raise ValueError(f"unsupported backend: {self.config.backend}")

    def init(self) -> None:
        if self._rtl_backend is not None:
            self._rtl_backend.init()
            self.regs = self._rtl_backend.regs
            self.generated = []
            self.cache = KVCache(max_seq=self.config.max_seq, dim=self.config.dim)
            self.last_error = ""
            return

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
            rm.REG_CFG_K_TILE: 16,
        }
        self.generated = []
        self.cache = KVCache(max_seq=self.config.max_seq, dim=self.config.dim)
        self.last_error = ""

    def load(self, pack_dir: Path | str) -> None:
        if self._rtl_backend is not None:
            self._rtl_backend.load(pack_dir)
            return

        p = Path(pack_dir)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        if int(meta["dim"]) != self.config.dim:
            raise ValueError("pack dim mismatch")
        self.weights["w_q"] = np.load(p / "w_q_int8.npy")
        self.weights["w_k"] = np.load(p / "w_k_int8.npy")
        self.weights["w_v"] = np.load(p / "w_v_int8.npy")
        self.weights["dequant_scale"] = np.array([meta["dequant_scale"]], dtype=np.float32)

    def run(self, prompt_tokens: np.ndarray, gen_len: int) -> np.ndarray:
        if self._rtl_backend is not None:
            out = self._rtl_backend.run(prompt_tokens=prompt_tokens, gen_len=gen_len)
            self.regs = self._rtl_backend.regs
            self.generated = [o for o in out]
            return out

        try:
            self.regs[rm.REG_CONTROL] = rm.CTRL_START
            self.regs[rm.REG_STATUS] = rm.STATUS_BUSY
            self.regs[rm.REG_PROMPT_LEN] = int(prompt_tokens.shape[0])
            self.regs[rm.REG_GEN_LEN] = int(gen_len)
            self.regs[rm.REG_DONE_TOKENS] = 0
            self.regs[rm.REG_PERF_CYCLES] = 0
            self.regs[rm.REG_PERF_TOKENS] = 0
            self.regs[rm.REG_PERF_STALL_IN] = 0
            self.regs[rm.REG_PERF_STALL_OUT] = 0
            self.regs[rm.REG_LAST_ERROR] = 0

            if prompt_tokens.ndim != 2 or prompt_tokens.shape[1] != self.config.dim:
                raise ValueError("prompt shape must be [T, D]")

            x_t = prompt_tokens[-1].astype(np.int16)
            scale = float(self.weights["dequant_scale"][0])

            outputs = []
            for _ in range(gen_len):
                q = gemm_int8w_int16a_acc32(x_t.reshape(1, -1), self.weights["w_q"]).reshape(-1).astype(np.float32)
                k = gemm_int8w_int16a_acc32(x_t.reshape(1, -1), self.weights["w_k"]).reshape(-1).astype(np.float32)
                v = gemm_int8w_int16a_acc32(x_t.reshape(1, -1), self.weights["w_v"]).reshape(-1).astype(np.float32)

                self.cache.append(k, v)
                k_all, v_all = self.cache.get()
                y = attention_decode_step(q, k_all, v_all)

                y_int16 = requantize_int16(np.round(y).astype(np.int32), scale=scale)
                outputs.append(y_int16.copy())
                x_t = y_int16
                self.regs[rm.REG_DONE_TOKENS] += 1
                self.regs[rm.REG_PERF_TOKENS] += 1
                # Numpy backend keeps perf counters minimal and deterministic.
                self.regs[rm.REG_PERF_CYCLES] += int(max(1, self.config.dim // 2))

            out = np.stack(outputs, axis=0)
            self.generated = [o for o in outputs]
            self.regs[rm.REG_STATUS] = rm.STATUS_DONE
            return out
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.regs[rm.REG_STATUS] = rm.STATUS_ERROR
            raise

    def poll(self) -> dict[str, int | str]:
        if self._rtl_backend is not None:
            return self._rtl_backend.poll()

        return {
            "status": self.regs.get(rm.REG_STATUS, 0),
            "done_tokens": self.regs.get(rm.REG_DONE_TOKENS, 0),
            "prompt_len": self.regs.get(rm.REG_PROMPT_LEN, 0),
            "gen_len": self.regs.get(rm.REG_GEN_LEN, 0),
            "perf_cycles": self.regs.get(rm.REG_PERF_CYCLES, 0),
            "perf_tokens": self.regs.get(rm.REG_PERF_TOKENS, 0),
            "perf_stall_in": self.regs.get(rm.REG_PERF_STALL_IN, 0),
            "perf_stall_out": self.regs.get(rm.REG_PERF_STALL_OUT, 0),
            "backend": "numpy",
        }
