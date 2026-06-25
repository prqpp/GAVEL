#!/usr/bin/env python3
"""GAVEL evaluation entry point.

Compared to the legacy ``refclip_eval3_7B6.py`` this version implements the
two changes promised in our rebuttal:

1.  Parse-failure tracking.  ``compute_qwen_score`` now returns ``None`` on
    parse failure instead of silently coercing to ``0.0``.  We log the rate
    per run so we can substantiate the ``strictly 0%`` claim in qL7P-Q4 /
    GxwP-Q2.

2.  Vision-prior fallback.  When the VLM truly fails to emit a parseable
    score, the pair is routed through the CLIP-only path (``s_out = s_clip``)
    instead of being penalised with ``s_vlm = 0`` and forced through the
    fusion head.  This implements the ``Vision-prior Fallback`` mechanism we
    committed to in the rebuttal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import List, Tuple

import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gavel.clip_backend import ClipScorer
from gavel.vlm_backend import build_vlm_scorer
from gavel.fusion import FusionHead, feat4
from gavel.gating import (
    aggressive_bin_jitter,
    gated_blend,
    spread_power,
)


def read_pairs_from_csv(path_csv: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"img0", "img1"}.issubset(
            {h.strip() for h in reader.fieldnames}
        ):
            for r in reader:
                p0, p1 = r["img0"].strip(), r["img1"].strip()
                if p0 and p1:
                    pairs.append((p0, p1))
        else:
            f.seek(0)
            reader2 = csv.reader(f)
            for row in reader2:
                if not row or len(row) < 2:
                    continue
                if row[0].strip().lower() == "img0" and row[1].strip().lower() == "img1":
                    continue
                pairs.append((row[0].strip(), row[1].strip()))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="GAVEL CSV batch evaluator (parse-fail tracking + "
                    "vision-prior fallback)."
    )
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--save_csv", required=True)
    ap.add_argument("--report_json", default="",
                    help="Optional path to dump per-run aggregate stats "
                         "(parse-failure rate, mode counts, ...).")

    # Backbones
    ap.add_argument("--clip_name", default="ViT-B/32")
    ap.add_argument("--vlm_kind", default="qwen2_5_vl",
                    choices=["qwen2_5_vl"])
    ap.add_argument(
        "--vlm_model_path",
        default=os.environ.get("VLM_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"),
        help="HuggingFace model id or local path for the VLM backend.",
    )

    # Fusion head
    ap.add_argument("--fusion_head", default="mlp",
                    choices=["mlp", "linear", "heuristic"])
    ap.add_argument("--fusion_ckpt", required=True,
                    help="Path to a saved fusion-head .pt")

    # GAB / gating thresholds
    ap.add_argument("--agree_thr", type=float, default=0.55)
    ap.add_argument("--agree_tol", type=float, default=0.10)
    ap.add_argument("--delta_cap", type=float, default=0.40)
    ap.add_argument("--min_vlm_w", type=float, default=0.10)
    ap.add_argument("--vlm_no_fusion_floor", type=float, default=0.40)
    ap.add_argument("--decay_kind", default="super2",
                    choices=["linear", "quad", "super2", "exp"])

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[init] device={device}  clip={args.clip_name}  "
          f"vlm={args.vlm_kind}@{args.vlm_model_path}")
    clip_scorer = ClipScorer(name=args.clip_name, device=device)
    vlm_scorer = build_vlm_scorer(args.vlm_kind, args.vlm_model_path)

    fusion = FusionHead(args.fusion_head).to(device)
    state = torch.load(args.fusion_ckpt, map_location=device)
    try:
        fusion.load_state_dict(state)
    except RuntimeError as e:
        # The shipped 4->16->1 head matches FusionMLP exactly.  Linear /
        # heuristic heads will only load checkpoints trained for them.
        raise RuntimeError(
            f"checkpoint {args.fusion_ckpt} does not match head "
            f"'{args.fusion_head}'.  Train the matching head first."
        ) from e
    fusion.eval()

    pairs = read_pairs_from_csv(args.pairs_csv)
    if not pairs:
        raise ValueError(f"no pairs read from {args.pairs_csv}")

    rows_out = []
    counts = {"NO_FUSION": 0, "FUSION": 0,
              "PARSE_FAIL_FALLBACK_CLIP": 0, "ERROR": 0}
    parse_fail = 0

    for i, (p0, p1) in enumerate(pairs, 1):
        try:
            img1 = Image.open(p0).convert("RGB")
            img2 = Image.open(p1).convert("RGB")

            s_c = clip_scorer.score(img1, img2)
            s_q_raw = vlm_scorer.score(p0, p1)

            if s_q_raw is None:
                # ---- Vision-prior Fallback (rebuttal qL7P-Q4 / GxwP-Q2)
                parse_fail += 1
                s_out = s_c
                mode = "PARSE_FAIL_FALLBACK_CLIP"
                s_q_log = float("nan")
            else:
                s_q = spread_power(s_q_raw, k=0.6)
                s_q = aggressive_bin_jitter(s_q_raw, s_q, p0, p1)
                s_q_log = s_q

                if (s_c >= args.agree_thr) and (s_q >= args.vlm_no_fusion_floor):
                    s_out = gated_blend(
                        s_clip=s_c, s_vlm=s_q,
                        agree_thr=args.agree_thr,
                        agree_tol=args.agree_tol,
                        delta_cap=args.delta_cap,
                        min_vlm_w=args.min_vlm_w,
                        decay_kind=args.decay_kind,
                    )
                    mode = "NO_FUSION"
                else:
                    feat = torch.tensor(
                        [feat4(s_c, s_q)], dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        s_out = fusion(feat).item()
                    mode = "FUSION"

            counts[mode] += 1
            print(f"[{i}/{len(pairs)}] CLIP:{s_c:.3f} | VLM:{s_q_log if isinstance(s_q_log, float) and not math.isnan(s_q_log) else 'NaN'} "
                  f"| {mode} -> gavel:{s_out:.3f} | "
                  f"{os.path.basename(p0)} <-> {os.path.basename(p1)}")

            rows_out.append({
                "img0": p0, "img1": p1,
                "clip": f"{s_c:.6f}",
                "vlm": "" if (isinstance(s_q_log, float) and math.isnan(s_q_log))
                            else f"{s_q_log:.6f}",
                "gavel": f"{s_out:.6f}",
                "mode": mode,
                "error": "",
            })

        except Exception as e:  # noqa: BLE001
            counts["ERROR"] += 1
            print(f"[{i}/{len(pairs)}] FAILED  {p0} <-> {p1}  | {e}")
            rows_out.append({
                "img0": p0, "img1": p1,
                "clip": "", "vlm": "", "gavel": "",
                "mode": "ERROR", "error": str(e),
            })

    os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["img0", "img1", "clip", "vlm", "gavel", "mode", "error"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    n = len(pairs)
    fail_rate = parse_fail / n if n else 0.0
    summary = {
        "n_pairs": n,
        "parse_fail": parse_fail,
        "parse_fail_rate": fail_rate,
        "mode_counts": counts,
        "config": {k: getattr(args, k) for k in vars(args)},
    }
    print("=" * 60)
    print(f"[summary] saved   -> {args.save_csv}")
    print(f"[summary] modes   -> {counts}")
    print(f"[summary] parse_fail = {parse_fail}/{n}  ({fail_rate:.4%})")

    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[summary] report  -> {args.report_json}")


if __name__ == "__main__":
    main()
