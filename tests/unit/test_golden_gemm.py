from __future__ import annotations

import numpy as np
import pytest

from tests.golden.golden_ops import (
    clamp_int8,
    clamp_int16,
    gemm_int8w_int16a_acc32,
    requantize_to_int16,
)


@pytest.mark.parametrize(
    "m,k,n,seed",
    [
        (1, 4, 1, 0),
        (1, 8, 8, 1),
        (2, 16, 8, 2),
        (4, 16, 4, 3),
        (8, 16, 8, 4),
        (8, 32, 16, 5),
        (16, 32, 16, 6),
        (16, 64, 8, 7),
        (32, 32, 32, 8),
        (4, 64, 64, 9),
    ],
)
def test_gemm_matches_numpy_int32(m: int, k: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    a = clamp_int16(rng.integers(-1024, 1024, size=(m, k)))
    b = clamp_int8(rng.integers(-64, 64, size=(k, n)))
    bias = rng.integers(-1000, 1000, size=(n,), dtype=np.int32)

    out = gemm_int8w_int16a_acc32(a, b, bias)
    expected = (a.astype(np.int32) @ b.astype(np.int32)) + bias
    np.testing.assert_array_equal(out, expected.astype(np.int32))


def test_gemm_shape_mismatch_raises():
    a = np.zeros((2, 3), dtype=np.int16)
    b = np.zeros((4, 5), dtype=np.int8)
    with pytest.raises(ValueError):
        gemm_int8w_int16a_acc32(a, b)


def test_requantize_to_int16_range():
    x = np.array([-(2**31), -12345, 0, 12345, 2**31 - 1], dtype=np.int32)
    y = requantize_to_int16(x, scale_num=1, scale_den=1024, zero_point=0)
    assert y.dtype == np.int16
    assert np.max(y) <= 32767
    assert np.min(y) >= -32768
