# GAVEL — Gated Assessment via VLM-guided Evaluator for Look-alike pairs

> *"GAVEL: A Post-hoc, Plug-and-Play Framework for Mitigating LASD
> Hallucinations in CLIP-style Similarity Assessment"*.

GAVEL is a **post-hoc**, **plug-and-play** calibration framework that
suppresses CLIP's *Look-Alike-but-Semantically-Different* (LASD)
hallucinations by routing image pairs through a Vision-Language model
(default: Qwen2.5-VL-7B) acting as a **semantic gate**.  When the visual and
semantic signals agree (a "safe zone" delimited by GAB), the framework
applies a smooth blend; when they disagree, it triggers a lightweight
learnable fusion head that has been trained on a small human-preference
dataset.
---

## 1. Repository layout

```
GAVEL/
├── README.md                 ← this file
├── requirements.txt
├── configs/
│   ├── qwen2_5_vl_7b.yaml    ← default
│   ├── qwen2_5_vl_32b.yaml
│   └── internvl2.yaml        ← cross-VLM ablation (interface stub)
├── gavel/
│   ├── clip_backend.py       ← CLIP wrapper
│   ├── vlm_backend.py        ← pluggable VLM backends (Qwen2.5-VL, InternVL2, …)
│   ├── parsing.py            ← robust score parsing (returns None on failure)
│   ├── fusion.py             ← FusionHead{mlp,linear,heuristic}
│   ├── gating.py             ← GAB + decay variants {linear,quad,super2,exp}
│   └── qwen_vl_utils.py
├── scripts/
│   ├── eval.py               ← inference / evaluation entry
│   ├── train_fusion.py       ← train the fusion head with sanitisation
│   ├── significance.py       ← mean ± std + paired Wilcoxon across splits
│   └── reproduce_main_table.sh
├── data/
│   └── README.md             ← documents the expected CSV manifests
├── checkpoints/
│   └── .gitkeep              ← put local/private fusion checkpoints here
└── legacy/                   ← kept empty here; use the original RefClip/ for archive
```

---

## 2. Models

The code uses the following models/backbones:

| Role | Default | Where configured |
|---|---|---|
| Visual encoder | OpenAI CLIP `ViT-B/32` | `--clip_name` / `configs/*.yaml` |
| Semantic gate VLM | `Qwen/Qwen2.5-VL-7B-Instruct` | `--vlm_model_path`, `VLM_PATH`, `configs/qwen2_5_vl_7b.yaml` |
| 32B VLM ablation | `Qwen/Qwen2.5-VL-32B-Instruct` | `configs/qwen2_5_vl_32b.yaml` |
| Cross-VLM ablation placeholder | `OpenGVLab/InternVL2-8B` | `configs/internvl2.yaml` |
| Fusion head | `FusionMLP` (`4 -> 16 -> 1`) | `checkpoints/refclip_fusion.pt` |

Fusion-head checkpoints are not committed by default. Put your local
`refclip_fusion.pt` under `checkpoints/`, or train a new one with
`scripts/train_fusion.py`. Qwen, InternVL2, CLIP, and dataset files are not
vendored in this repository.

---

## 3. Install

```bash
conda create -n gavel python=3.10 -y && conda activate gavel
pip install -r requirements.txt
```

You will additionally need:

* a Qwen2.5-VL-7B-Instruct checkpoint reachable by HuggingFace model id or local path
* an OpenAI CLIP `ViT-B/32` weight (auto-downloaded on first use)

---

## 4. Quick start

### Inference

```bash
python scripts/eval.py \
    --pairs_csv  ./data/pairs_split1.csv \
    --save_csv   ./outputs/eval/out_gavel_split_1.csv \
    --report_json ./outputs/reports/report_gavel_split_1.json \
    --vlm_model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --fusion_ckpt   ./checkpoints/refclip_fusion.pt
```

This writes per-pair predictions plus an aggregate JSON containing the
**parse-failure rate**, the count of pairs routed through the fusion head
vs. the smooth blend path, and the exact configuration used.

### Training the fusion head

```bash
python scripts/train_fusion.py \
    --train_csv ./data/train_pairs.csv \
    --out_ckpt  ./checkpoints/refclip_fusion.pt \
    --out_features_csv ./outputs/train_features_cache.csv \
    --fusion_head mlp     # or `linear` / `heuristic`
```

### Statistical significance across splits

```bash
python scripts/significance.py \
    --summary_glob "./outputs/corr/corr_summary_split_*.csv" \
    --reference   out_gavel_split \
    --out_csv     ./outputs/significance_table.csv
```

### One-stop reproduction

```bash
bash scripts/reproduce_main_table.sh
```

---


## 5. Hyper-parameters at a glance

| Symbol | Default | Description |
|---|---|---|
| `agree_thr` (τ₁) | 0.55 | CLIP threshold above which the pair *may* skip the fusion head. |
| `vlm_no_fusion_floor` (τ₂) | 0.40 | VLM threshold; both τ₁ AND τ₂ must be exceeded to skip fusion. |
| `agree_tol` | 0.10 | If the two signed deviations agree within this tolerance, output the simple mean. |
| `delta_cap` | 0.40 | Saturation cap for the disagreement gap before computing the decay weight. |
| `min_vlm_w` | 0.10 | Lower bound on the VLM weight in the smooth-blend path. |
| `decay_kind` | `super2` | Decay function for the smooth-blend weight. `super2` corresponds to (1−x)^2.5 and is the paper's default. |

The asymmetry of (τ₁, τ₂) is intentional — see Sec. 3.3 of the paper for
why CLIP and VLM scores need different operating points.

---
<!--
## 6. Citation

```bibtex
@inproceedings{gavel2026,
  title  = {GAVEL: A Post-hoc, Plug-and-Play Framework for Mitigating LASD
            Hallucinations in CLIP-style Similarity Assessment},
  author = {Anonymous},
  booktitle = {ICML},
  year   = {2026}
}
```
-->
## 6. License

Released for academic research use. Replace this note with the final project
license before public archival release if your venue or institution requires a
specific license.
