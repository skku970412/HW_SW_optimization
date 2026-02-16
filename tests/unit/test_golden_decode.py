from __future__ import annotations

import numpy as np

from tests.golden.golden_decode import decode_step_reference
from tests.golden.golden_kvcache import KVCache
from tests.golden.golden_ops import clamp_int8, clamp_int16


def test_decode_step_updates_cache_length():
    d_model = 16
    rng = np.random.default_rng(10)
    x_t = clamp_int16(rng.integers(-512, 512, size=(d_model,)))
    w_q = clamp_int8(rng.integers(-8, 8, size=(d_model, d_model)))
    w_k = clamp_int8(rng.integers(-8, 8, size=(d_model, d_model)))
    w_v = clamp_int8(rng.integers(-8, 8, size=(d_model, d_model)))
    cache = KVCache(max_seq=8, num_heads=1, head_dim=d_model)

    out1 = decode_step_reference(x_t, w_q, w_k, w_v, cache, scale_num=1, scale_den=16)
    out2 = decode_step_reference(x_t, w_q, w_k, w_v, cache, scale_num=1, scale_den=16)

    assert out1.shape == (d_model,)
    assert out2.shape == (d_model,)
    assert out1.dtype == np.int16
    assert out2.dtype == np.int16
    assert cache.length == 2
