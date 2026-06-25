"""Thin wrapper around OpenAI CLIP for the visual half of GAVEL."""

from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image

import clip


class ClipScorer:
    def __init__(self, name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, img1: Image.Image, img2: Image.Image) -> float:
        t1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        t2 = self.preprocess(img2).unsqueeze(0).to(self.device)
        f1 = self.model.encode_image(t1)
        f2 = self.model.encode_image(t2)
        f1 /= f1.norm(dim=-1, keepdim=True)
        f2 /= f2.norm(dim=-1, keepdim=True)
        return (f1 @ f2.T).item()
