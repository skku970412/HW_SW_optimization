from __future__ import annotations

import numpy as np


def softmax_exact(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom


def exp_approx_piecewise(x: np.ndarray) -> np.ndarray:
    """
    LUT-based linear interpolation for exp(x) on [-8, 0].
    For x > 0 clamp to 0, for x < -8 clamp to -8.
    """
    x_clip = np.clip(x, -8.0, 0.0)
    lut_x = np.linspace(-8.0, 0.0, 257, dtype=np.float32)
    lut_y = np.exp(lut_x)
    pos = ((x_clip + 8.0) / 8.0) * (len(lut_x) - 1)
    idx0 = np.floor(pos).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, len(lut_x) - 1)
    frac = pos - idx0
    return lut_y[idx0] * (1.0 - frac) + lut_y[idx1] * frac


def softmax_approx(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = exp_approx_piecewise(x - x_max)
    exp = np.maximum(exp, 1e-8)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
    use_approx_softmax: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    q: [Tq, Dh], k: [Tk, Dh], v: [Tk, Dh]
    returns:
      output: [Tq, Dh]
      probs:  [Tq, Tk]
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, v must be 2D")
    if k.shape[0] != v.shape[0] or q.shape[1] != k.shape[1] or v.shape[1] != q.shape[1]:
        raise ValueError("attention shape mismatch")

    dh = q.shape[1]
    scale = 1.0 / np.sqrt(float(dh))
    scores = (q @ k.T) * scale

    if causal:
        tq, tk = scores.shape
        mask = np.triu(np.ones((tq, tk), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)

    probs = softmax_approx(scores, axis=-1) if use_approx_softmax else softmax_exact(scores, axis=-1)
    output = probs @ v
    return output, probs
