from __future__ import annotations

import numpy as np
import pytest

from tests.golden.golden_kvcache import KVCache


def test_kvcache_append_and_read_prefix():
    cache = KVCache(max_seq=4, num_heads=2, head_dim=3)
    for i in range(3):
        k = np.full((2, 3), i + 1, dtype=np.float32)
        v = np.full((2, 3), (i + 1) * 10, dtype=np.float32)
        cache.append(k, v)
    k_all, v_all = cache.get_prefix()
    assert k_all.shape == (3, 2, 3)
    assert v_all.shape == (3, 2, 3)
    assert cache.length == 3


def test_kvcache_overflow_raises():
    cache = KVCache(max_seq=1, num_heads=1, head_dim=2)
    cache.append(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        cache.append(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32))


def test_kvcache_shape_mismatch_raises():
    cache = KVCache(max_seq=2, num_heads=2, head_dim=4)
    with pytest.raises(ValueError):
        cache.append(np.zeros((1, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))
