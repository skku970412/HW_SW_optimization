from __future__ import annotations

import numpy as np


def clamp_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767).astype(np.int16)


def clamp_int8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -128, 127).astype(np.int8)


def gemm_int8w_int16a_acc32(
    a_int16: np.ndarray,
    b_int8: np.ndarray,
    bias_int32: np.ndarray | None = None,
) -> np.ndarray:
    """
    Quantized GEMM reference:
    - A: [M, K] int16 activations
    - B: [K, N] int8 weights
    - Output: [M, N] int32 accumulations
    """
    if a_int16.ndim != 2 or b_int8.ndim != 2:
        raise ValueError("a_int16 and b_int8 must be 2D tensors")
    m, k_a = a_int16.shape
    k_b, n = b_int8.shape
    if k_a != k_b:
        raise ValueError(f"GEMM shape mismatch: A K={k_a}, B K={k_b}")

    out = a_int16.astype(np.int32) @ b_int8.astype(np.int32)
    if bias_int32 is not None:
        if bias_int32.shape != (n,):
            raise ValueError(f"bias shape must be ({n},)")
        out = out + bias_int32.astype(np.int32)
    return out.astype(np.int32)


def requantize_to_int16(
    x_int32: np.ndarray, scale_num: int, scale_den: int, zero_point: int = 0
) -> np.ndarray:
    """
    Integer-friendly requantization:
    y = round(x * scale_num / scale_den) + zero_point
    """
    if scale_den <= 0:
        raise ValueError("scale_den must be positive")
    scaled = np.round((x_int32.astype(np.float64) * scale_num) / scale_den)
    shifted = scaled + zero_point
    return clamp_int16(shifted)


def relu_int16(x_int16: np.ndarray) -> np.ndarray:
    return np.maximum(x_int16, 0).astype(np.int16)
