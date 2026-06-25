#!/usr/bin/env python3
"""Train the GAVEL fusion head.

Two changes vs. the legacy ``train_fusion_7B.py``:

1.  Training-phase sanitisation (rebuttal qL7P-Q4 / GxwP-Q2 / GxwP-W4).
    Image pairs whose VLM score is ``0.0`` (treated as a parse-failure
    sentinel) are dropped from the training batch, so the MLP never learns to
    impose a severe non-linear penalty on legitimate ``s_vlm = 0`` pairs.  We
    log the number of dropped rows so the manuscript can quote a real number.

2.  Pluggable head.  ``--fusion_head {mlp,linear,heuristic}`` lets us produce
    the fusion-ablation table (Logistic Regression / closed-form heuristic /
    MLP) requested by reviewer GxwP-W1.

The CSV format is:

    img0_path,img1_path,label[,precomputed_clip,precomputed_vlm]

If the optional precomputed columns are present we skip the (slow) CLIP+VLM
forward passes and use them directly.  Otherwise we run the backbones.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gavel.clip_backend import ClipScorer
from gavel.fusion import FusionHead, feat4
from gavel.vlm_backend import build_vlm_scorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(csv_path: str
              ) -> List[Tuple[str, str, float, Optional[float], Optional[float]]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            # tolerate header
            if idx == 1 and row[0].lower() in {"img0", "img1", "image0", "path0"}:
                continue
            if len(row) < 3:
                print(f"[skip line {idx}] need >=3 columns: {row}", file=sys.stderr)
                continue
            p0, p1, lab = row[0].strip(), row[1].strip(), row[2].strip()
            if not p0 or not p1:
                continue
            try:
                label = float(lab)
            except ValueError:
                print(f"[skip line {idx}] bad label {lab!r}", file=sys.stderr)
                continue
            s_clip = _maybe_float(row[3]) if len(row) > 3 else None
            s_vlm = _maybe_float(row[4]) if len(row) > 4 else None
            out.append((p0, p1, label, s_clip, s_vlm))
    return out


def compute_features(rows, args, device: str
                     ) -> Tuple[List[List[float]], List[float], dict]:
    """Compute (or read precomputed) (s_clip, s_vlm) for every row.

    Returns (features, labels, stats) where stats contains the parse-failure
    sanitisation counters expected by the rebuttal text.
    """
    need_backbone = any(r[3] is None or r[4] is None for r in rows)

    clip_scorer = None
    vlm_scorer = None
    if need_backbone:
        print("[init] building CLIP + VLM (some rows have no precomputed scores)")
        clip_scorer = ClipScorer(name=args.clip_name, device=device)
        vlm_scorer = build_vlm_scorer(args.vlm_kind, args.vlm_model_path)

    features: List[List[float]] = []
    labels: List[float] = []
    n_total = 0
    n_dropped_sanitize = 0          # s_vlm == 0.0 (parse-fail sentinel)
    n_dropped_missing_file = 0
    n_dropped_parse_fail_live = 0   # live VLM call returned None

    for (p0, p1, lab, s_c, s_v) in rows:
        n_total += 1

        if not (os.path.exists(p0) and os.path.exists(p1)):
            n_dropped_missing_file += 1
            continue

        if s_c is None or s_v is None:
            img1 = Image.open(p0).convert("RGB")
            img2 = Image.open(p1).convert("RGB")
            if s_c is None:
                s_c = clip_scorer.score(img1, img2)
            if s_v is None:
                s_v_live = vlm_scorer.score(p0, p1)
                if s_v_live is None:
                    n_dropped_parse_fail_live += 1
                    continue
                s_v = s_v_live

        # ---- Training-phase sanitisation (rebuttal commitment) -----------
        # The legacy pipeline emitted s_vlm=0.0 as a parse-failure sentinel.
        # Such rows are dropped so the head never learns to penalise
        # legitimate s_vlm=0 pairs.
        if args.sanitize_zero_vlm and abs(s_v) < 1e-9:
            n_dropped_sanitize += 1
            continue
        # ------------------------------------------------------------------

        features.append(feat4(s_c, s_v))
        labels.append(lab)

    stats = {
        "n_total_rows": n_total,
        "n_kept": len(features),
        "n_dropped_sanitize_zero_vlm": n_dropped_sanitize,
        "n_dropped_missing_file": n_dropped_missing_file,
        "n_dropped_parse_fail_live": n_dropped_parse_fail_live,
    }
    return features, labels, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--out_features_csv", default="",
                    help="Optional cache of (s_clip, s_vlm, label) so "
                         "subsequent fusion-head ablations can reuse them.")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--fusion_head", default="mlp",
                    choices=["mlp", "linear", "heuristic"])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--clip_name", default="ViT-B/32")
    ap.add_argument("--vlm_kind", default="qwen2_5_vl")
    ap.add_argument(
        "--vlm_model_path",
        default=os.environ.get("VLM_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"),
        help="HuggingFace model id or local path for the VLM backend.",
    )

    ap.add_argument("--sanitize_zero_vlm", action="store_true", default=True,
                    help="Drop rows where s_vlm==0.0 (parse-fail sentinel). "
                         "ON by default; pass --no-sanitize_zero_vlm to disable.")
    ap.add_argument("--no-sanitize_zero_vlm", dest="sanitize_zero_vlm",
                    action="store_false")

    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_rows(args.train_csv)
    print(f"[data] loaded {len(rows)} candidate rows from {args.train_csv}")

    feats, labs, stats = compute_features(rows, args, device)
    print("=" * 60)
    print("[sanitise]")
    for k, v in stats.items():
        print(f"  {k:>32s} : {v}")
    print("=" * 60)
    if not feats:
        raise RuntimeError("No training rows survived sanitisation.")

    if args.out_features_csv:
        os.makedirs(os.path.dirname(args.out_features_csv) or ".", exist_ok=True)
        with open(args.out_features_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["s_clip", "s_vlm", "abs_diff", "prod", "label"])
            for fr, lb in zip(feats, labs):
                w.writerow(fr + [lb])
        print(f"[cache] feature cache -> {args.out_features_csv}")

    X = torch.tensor(feats, dtype=torch.float32, device=device)
    Y = torch.tensor([[v] for v in labs], dtype=torch.float32, device=device)

    head = FusionHead(args.fusion_head, hidden=args.hidden) \
        if args.fusion_head == "mlp" else FusionHead(args.fusion_head)
    head = head.to(device)

    opt = optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    head.train()
    for epoch in range(1, args.epochs + 1):
        pred = head(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"[train] epoch {epoch:>4d}/{args.epochs}  loss={loss.item():.5f}")

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(head.state_dict(), args.out_ckpt)
    print(f"[done] head ({args.fusion_head}) saved -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
