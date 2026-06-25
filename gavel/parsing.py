"""Robust parsing utilities for VLM textual outputs.

Returns ``None`` when nothing parseable is found so callers can apply a
vision-prior fallback (do NOT silently coerce to 0.0; see rebuttal commitment
to ``Vision-prior Fallback`` in response to qL7P-Q4 / GxwP-Q2).
"""

from __future__ import annotations

import json
import re
from typing import Optional


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", flags=re.S)
_CODEBLOCK_RE = re.compile(r"```json(.*?)```|```(.*?)```", flags=re.S | re.I)
_DECIMAL_RE = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)")
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%")


def _last_json_block(text: str) -> Optional[str]:
    blocks = _JSON_BLOCK_RE.findall(text)
    return blocks[-1] if blocks else None


def _last_json_in_codeblock(text: str) -> Optional[str]:
    for cb in reversed(_CODEBLOCK_RE.findall(text)):
        body = cb[0] if cb[0] else cb[1]
        if not body:
            continue
        j = _last_json_block(body)
        if j:
            return j
    return None


def _try_parse_obj(j: str) -> Optional[float]:
    try:
        obj = json.loads(j.replace("：", ":").replace("'", '"'))
    except Exception:
        return None
    for k in ("score", "similarity", "相似度", "分数"):
        if k in obj:
            try:
                return float(obj[k])
            except Exception:
                continue
    return None


def parse_score_from_text(resp: str) -> Optional[float]:
    """Try multiple strategies, returning ``None`` on failure (NOT 0.0)."""
    j = _last_json_in_codeblock(resp)
    if j is not None:
        v = _try_parse_obj(j)
        if v is not None:
            return v

    j = _last_json_block(resp)
    if j is not None:
        v = _try_parse_obj(j)
        if v is not None:
            return v

    decimals = _DECIMAL_RE.findall(resp)
    if decimals:
        try:
            return float(decimals[-1])
        except Exception:
            pass

    percents = _PERCENT_RE.findall(resp)
    if percents:
        try:
            return float(percents[-1]) / 100.0
        except Exception:
            pass

    return None
