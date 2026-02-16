from __future__ import annotations

import numpy as np


def gemm_int8w_int16a_acc32(a_int16: np.ndarray, b_int8: np.ndarray) -> np.ndarray:
    if a_int16.ndim != 2 or b_int8.ndim != 2:
        raise ValueError("a_int16 and b_int8 must be 2D")
    if a_int16.shape[1] != b_int8.shape[0]:
        raise ValueError("gemm shape mismatch")
    return (a_int16.astype(np.int32) @ b_int8.astype(np.int32)).astype(np.int32)


def requantize_int16(x_int32: np.ndarray, scale: float) -> np.ndarray:
    out = np.round(x_int32.astype(np.float64) * scale)
    out = np.clip(out, -32768, 32767)
    return out.astype(np.int16)


class KVCache:
    def __init__(self, max_seq: int, dim: int) -> None:
        self.max_seq = max_seq
        self.dim = dim
        self.k = np.zeros((max_seq, dim), dtype=np.float32)
        self.v = np.zeros((max_seq, dim), dtype=np.float32)
        self.length = 0

    def append(self, k_t: np.ndarray, v_t: np.ndarray) -> None:
        if self.length >= self.max_seq:
            raise ValueError("kv overflow")
        if k_t.shape != (self.dim,) or v_t.shape != (self.dim,):
            raise ValueError("kv shape mismatch")
        self.k[self.length] = k_t
        self.v[self.length] = v_t
        self.length += 1

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self.k[: self.length], self.v[: self.length]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_decode_step(q_t: np.ndarray, k_all: np.ndarray, v_all: np.ndarray) -> np.ndarray:
    # q_t: [D], k_all/v_all: [T, D]
    scale = 1.0 / np.sqrt(float(q_t.shape[0]))
    score = (k_all @ q_t) * scale  # [T]
    prob = softmax(score.reshape(1, -1), axis=-1).reshape(-1)
    out = prob @ v_all
    return out.astype(np.float32)
