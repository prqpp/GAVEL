"""Fusion-head variants used in the GAVEL ablation table.

Default (``mlp``) reproduces the published checkpoint shipped as
``checkpoints/refclip_fusion.pt``.  The ``linear`` and ``heuristic`` options
are lightweight ablation heads.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def feat4(s_clip: float, s_vlm: float) -> List[float]:
    """4-dim hand-crafted feature used by all heads."""
    return [s_clip, s_vlm, abs(s_clip - s_vlm), s_clip * s_vlm]


class FusionMLP(nn.Module):
    """Backwards-compatible alias for the original published head (4 -> 16 -> 1)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.act(self.fc2(h))


class FusionLinear(nn.Module):
    """Logistic regression on the 4-dim hand-crafted feature."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class FusionHeuristic(nn.Module):
    """Closed-form ``alpha * s_vlm + (1-alpha) * s_clip``; alpha is learnable."""

    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(init_alpha / (1 - init_alpha)))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_clip = x[..., 0:1]
        s_vlm = x[..., 1:2]
        return self.alpha * s_vlm + (1.0 - self.alpha) * s_clip


def FusionHead(kind: str = "mlp", **kw) -> nn.Module:
    """Factory; ``kind in {mlp, linear, heuristic}``."""
    kind = kind.lower()
    if kind == "mlp":
        return FusionMLP(**kw)
    if kind == "linear":
        return FusionLinear()
    if kind == "heuristic":
        return FusionHeuristic(**kw)
    raise ValueError(f"unknown fusion head: {kind}")
