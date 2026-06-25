# GAVEL — Gated Assessment via VLM-guided Evaluator for Look-alike pairs

> Anonymous reference implementation accompanying the ICML 2026 submission
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

The codebase is intentionally small: the entire framework consists of
< 700 lines of Python, the fusion head has < 100 trainable parameters, and
both CLIP and the VLM are kept frozen.

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

## 5. Mapping the code onto the rebuttal commitments

The rebuttal mentions several behavioural commitments.  This release
materialises every one of them:

| Reviewer / point | Commitment | Where it lives |
|---|---|---|
| **qL7P-Q4 / GxwP-Q2** — *Vision-prior Fallback* | "In the rare event of a VLM parsing failure, GAVEL will safely degrade to the native CLIP baseline score." | `scripts/eval.py`: when `vlm_scorer.score(...)` returns `None`, `s_out := s_clip` and the mode is logged as `PARSE_FAIL_FALLBACK_CLIP`. |
| **qL7P-Q4 / GxwP-W4** — *Training-phase Sanitisation Flag* | "Image pairs marked with 0.0 are immediately intercepted and discarded from the training batch." | `scripts/train_fusion.py`: `--sanitize_zero_vlm` (ON by default) drops rows where `s_vlm = 0.0` *before* the head sees them.  The number of dropped rows is printed for transparency. |
| **qL7P-Q4 / GxwP-Q2** — *Parse-failure rate tracking* | "Across tens of thousands of evaluations the actual parsing failure rate of Qwen2.5-VL was strictly 0%." | `scripts/eval.py` aggregates and prints `parse_fail_rate` per run, and writes it into `--report_json` so the final manuscript can quote a verifiable number. |
| **qL7P-Q5** — *Super-quadratic decay ablation* | "We will provide an ablation of the penalty function design." | `gavel/gating.py::attenuate(kind=…)` exposes `linear / quad / super2 / exp`, switchable via `--decay_kind`. |
| **GxwP-W1** — *Fusion-head simplicity* | "MLP is needed for an extremely asymmetric, smooth penalty surface." | `gavel/fusion.py` ships three drop-in heads (`mlp`, `linear`, `heuristic`) sharing the same 4-D feature. |
| **GxwP-W3** — *N-expert generalisation* | "GAVEL generalises to N experts via a logical conjunction of thresholds." | The decoupling between `vlm_backend.py` and `fusion.py` keeps the semantic scorer pluggable. |
| **GxwP-Q1 / qL7P-Q2 / BSmU-Q3** — *Backbone-agnostic / plug-and-play* | InternVL2 / LLaVA cross-VLM evaluation. | `gavel/vlm_backend.py::VLMScorer` is an ABC; `configs/internvl2.yaml` documents the interface for an additional backbone. |
| **tUiR-W2** — *Statistical significance in the main table* | "We will integrate confidence intervals and standard deviations into the core tables." | `scripts/significance.py` aggregates the six existing `corr_summary_*.csv` files and emits a `mean ± std (Wilcoxon p)` table. |
| **tUiR-W3** — *Code release* | "We commit to fully open-sourcing the complete GAVEL codebase upon acceptance." | This repository is the artefact; reproduction is one shell script. |

---

## 6. Hyper-parameters at a glance

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

## 7. Citation

```bibtex
@inproceedings{gavel2026,
  title  = {GAVEL: A Post-hoc, Plug-and-Play Framework for Mitigating LASD
            Hallucinations in CLIP-style Similarity Assessment},
  author = {Anonymous},
  booktitle = {ICML},
  year   = {2026}
}
```

## 8. License

Released for academic research use. Replace this note with the final project
license before public archival release if your venue or institution requires a
specific license.
