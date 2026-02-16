from __future__ import annotations

import numpy as np

from tests.golden.golden_attention import scaled_dot_product_attention
from tests.golden.golden_kvcache import KVCache
from tests.golden.golden_ops import gemm_int8w_int16a_acc32, requantize_to_int16


def decode_step_reference(
    x_t_int16: np.ndarray,
    w_q_int8: np.ndarray,
    w_k_int8: np.ndarray,
    w_v_int8: np.ndarray,
    cache: KVCache,
    scale_num: int = 1,
    scale_den: int = 1,
) -> np.ndarray:
    """
    Single-token decode reference for one head group collapsed to [D].
    This is a lightweight MVP function for boardless tests.
    """
    if x_t_int16.ndim != 1:
        raise ValueError("x_t_int16 must be 1D")

    x2d = x_t_int16.reshape(1, -1)
    q = gemm_int8w_int16a_acc32(x2d, w_q_int8).astype(np.float32)
    k = gemm_int8w_int16a_acc32(x2d, w_k_int8).astype(np.float32)
    v = gemm_int8w_int16a_acc32(x2d, w_v_int8).astype(np.float32)

    # Treat as single-head for MVP reference.
    q_t = q.reshape(-1)
    k_t = k.reshape(-1)
    v_t = v.reshape(-1)
    cache.append(k_t.reshape(1, -1), v_t.reshape(1, -1))

    k_all, v_all = cache.get_prefix()  # [T, 1, D]
    out, _ = scaled_dot_product_attention(
        q_t.reshape(1, -1),
        k_all[:, 0, :],
        v_all[:, 0, :],
        causal=True,
        use_approx_softmax=True,
    )
    out_int32 = np.round(out).astype(np.int32)
    return requantize_to_int16(out_int32, scale_num=scale_num, scale_den=scale_den).reshape(-1)
