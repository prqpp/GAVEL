"""GAVEL: Gated Assessment via VLM-guided Evaluator for Look-alike pairs."""

__version__ = "1.0.0"

from .fusion import FusionHead
from .gating import gated_blend, attenuate
from .parsing import parse_score_from_text
