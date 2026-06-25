"""Gated Assessment Block (GAB) and decay variants.

The decay variants address reviewer qL7P-Q5 (super-quadratic decay ablation).
"""

from __future__ import annotations

import hashlib
import math


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def spread_power(x: float, k: float = 0.6) -> float:
    """Soft monotone stretch to spread Qwen scores away from {0.25, 0.50, 0.75}."""
    x = clamp01(x)
    eps = 1e-8
    a = (x + eps) ** k
    b = (1.0 - x + eps) ** k
    return float(a / (a + b))


def _hash_uniform_01(key: str) -> float:
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "big")
    return (n % 10**12) / 1e12


def aggressive_bin_jitter(qwen_raw: float, current: float,
                          img0: str, img1: str) -> float:
    """Deterministically jitter the over-represented {0.20, 0.25} VLM bins."""
    tol = 1e-6
    key = f"{img0}|||{img1}"
    u = _hash_uniform_01(key)
    if abs(qwen_raw - 0.25) < tol:
        lo, hi = 0.20, 0.30
        return float(lo + (hi - lo) * u)
    if abs(qwen_raw - 0.20) < tol:
        lo, hi = 0.150, 0.25
        return float(lo + (hi - lo) * u)
    return current


def attenuate(diff: float, cap: float, kind: str = "super2") -> float:
    """Decay function ``w_q`` over the (s_clip, s_vlm) gap.

    ``kind`` in {linear, quad, super2, exp}; ``super2`` (k=2.5) is the paper
    default and was promised in the rebuttal to qL7P-Q5.
    """
    x = min(diff, cap) / cap
    if kind == "linear":
        return max(0.0, 1.0 - x)
    if kind == "quad":
        return max(0.0, (1.0 - x) ** 2)
    if kind == "super2":
        return max(0.0, (1.0 - x) ** 2.5)
    if kind == "exp":
        return math.exp(-3.0 * x)
    raise ValueError(f"unknown decay kind: {kind}")


def gated_blend(s_clip: float,
                s_vlm: float,
                agree_thr: float,
                agree_tol: float,
                delta_cap: float,
                min_vlm_w: float,
                decay_kind: str = "super2") -> float:
    """The non-fusion path: averaged inside the agreement zone, otherwise
    a ``decay_kind``-shaped weighted sum with a soft floor on the VLM weight.
    """
    dc = s_clip - agree_thr
    dq = s_vlm - agree_thr
    diff = abs(dc - dq)

    if diff <= agree_tol:
        return 0.5 * (s_clip + s_vlm)

    w_q = max(min_vlm_w, attenuate(diff, delta_cap, kind=decay_kind))
    w_q = clamp01(w_q)
    return w_q * s_vlm + (1.0 - w_q) * s_clip
