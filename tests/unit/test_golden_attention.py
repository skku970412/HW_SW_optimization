from __future__ import annotations

import numpy as np
import pytest

from tests.golden.golden_attention import (
    scaled_dot_product_attention,
    softmax_approx,
    softmax_exact,
)


def test_softmax_exact_row_sum_close_to_one():
    x = np.array([[0.2, -0.1, 1.4], [1.1, 1.2, -3.3]], dtype=np.float32)
    y = softmax_exact(x)
    row_sum = np.sum(y, axis=-1)
    np.testing.assert_allclose(row_sum, np.ones_like(row_sum), atol=1e-6)


def test_softmax_approx_row_sum_close_to_one():
    x = np.array([[0.2, -0.1, 1.4], [1.1, 1.2, -3.3]], dtype=np.float32)
    y = softmax_approx(x)
    row_sum = np.sum(y, axis=-1)
    np.testing.assert_allclose(row_sum, np.ones_like(row_sum), atol=1e-4)


@pytest.mark.parametrize("tq,tk,dh,seed", [(1, 1, 8, 0), (2, 2, 8, 1), (4, 4, 16, 2), (8, 8, 32, 3)])
def test_attention_output_shape_and_probability_bounds(tq: int, tk: int, dh: int, seed: int):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(tq, dh)).astype(np.float32)
    k = rng.normal(size=(tk, dh)).astype(np.float32)
    v = rng.normal(size=(tk, dh)).astype(np.float32)
    out, probs = scaled_dot_product_attention(q, k, v, causal=True, use_approx_softmax=True)
    assert out.shape == (tq, dh)
    assert probs.shape == (tq, tk)
    assert np.all(probs >= 0.0)
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones((tq,)), atol=1e-3)


def test_approx_vs_exact_softmax_error_reasonable():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 16)).astype(np.float32)
    exact = softmax_exact(x)
    approx = softmax_approx(x)
    max_abs_err = np.max(np.abs(exact - approx))
    # Loose bound for cubic approximation-based MVP.
    assert max_abs_err < 0.2
