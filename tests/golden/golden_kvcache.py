from __future__ import annotations

import numpy as np


class KVCache:
    """
    Simple KV cache for boardless functional validation.
    Layout: [max_seq, num_heads, head_dim]
    """

    def __init__(self, max_seq: int, num_heads: int, head_dim: int) -> None:
        self.max_seq = max_seq
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k = np.zeros((max_seq, num_heads, head_dim), dtype=np.float32)
        self.v = np.zeros((max_seq, num_heads, head_dim), dtype=np.float32)
        self.length = 0

    def append(self, k_t: np.ndarray, v_t: np.ndarray) -> None:
        if self.length >= self.max_seq:
            raise ValueError("KV cache overflow")
        if k_t.shape != (self.num_heads, self.head_dim):
            raise ValueError("k_t shape mismatch")
        if v_t.shape != (self.num_heads, self.head_dim):
            raise ValueError("v_t shape mismatch")
        self.k[self.length] = k_t
        self.v[self.length] = v_t
        self.length += 1

    def get_prefix(self) -> tuple[np.ndarray, np.ndarray]:
        return self.k[: self.length], self.v[: self.length]
