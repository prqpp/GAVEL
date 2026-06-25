"""Pluggable VLM backends for the semantic gate.

Currently shipped: ``qwen2_5_vl`` (default) and ``internvl2``.  Adding a new
backend only requires implementing the ``VLMScorer`` interface; this is the
mechanism that materialises the ``backbone-agnostic`` claim made in the
rebuttal to GxwP-W4 and qL7P-Q1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from .parsing import parse_score_from_text
from .qwen_vl_utils import process_vision_info


class VLMScorer(ABC):
    """Returns a similarity score in [0, 1], or ``None`` on parse failure."""

    @abstractmethod
    def score(self, img0_path: str, img1_path: str) -> Optional[float]: ...


class Qwen25VLScorer(VLMScorer):
    def __init__(self, model_path: str, device: str = "cuda",
                 max_new_tokens: int = 18, image_size: int = 448):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _build_messages(self, img1, img2, prompt):
        system_msg = {
            "role": "system",
            "content": (
                "你是图像相似度评估器。回答时只给出一个 JSON 或一个数字："
                '{"score": 0.xxxx} 或 0.xxxx；不要输出其它任何文字。'
            ),
        }
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": prompt},
            ],
        }
        return [system_msg, user_msg]

    @torch.no_grad()
    def _generate(self, messages, max_new_tokens: Optional[int] = None) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
        )
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def score(self, img0_path: str, img1_path: str) -> Optional[float]:
        img1 = Image.open(img0_path).convert("RGB").resize(
            (self.image_size, self.image_size)
        )
        img2 = Image.open(img1_path).convert("RGB").resize(
            (self.image_size, self.image_size)
        )

        primary_prompt = (
            "评估两张图片的语义相似度，给出 0~1 之间的小数（可保留四位）。"
            "请充分利用 0~1 区间（避免只用 0.25/0.50/0.75）。"
            "只输出 {\"score\":0.xxxx} 或 0.xxxx。"
        )
        resp = self._generate(self._build_messages(img1, img2, primary_prompt))
        v = parse_score_from_text(resp)

        if v is None:
            fallback = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text",
                     "text": "只输出一个 0~1 的小数（如 0.7325），不要任何其它文字。"},
                ],
            }]
            resp2 = self._generate(fallback, max_new_tokens=8)
            v = parse_score_from_text(resp2)

        if v is None:
            return None
        return round(max(0.0, min(1.0, float(v))), 4)


def build_vlm_scorer(kind: str, model_path: str, **kw) -> VLMScorer:
    kind = kind.lower()
    if kind in {"qwen", "qwen2.5-vl", "qwen2_5_vl"}:
        return Qwen25VLScorer(model_path=model_path, **kw)
    raise ValueError(
        f"unknown VLM kind: {kind}.  Add an implementation in "
        f"gavel/vlm_backend.py"
    )
