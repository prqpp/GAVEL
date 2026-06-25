#!/usr/bin/env bash
# One-stop reproduction script for the GAVEL main table.
#
# Pre-requisites:
#   * A Qwen2.5-VL-7B-Instruct checkpoint accessible via HuggingFace, or a
#     local checkpoint path supplied with VLM_PATH.
#   * The trained fusion-head checkpoint at ./checkpoints/refclip_fusion.pt.
#   * Six pair-CSV splits at ./data/pairs_split{1..6}.csv with two columns
#     (img0, img1).
#   * A human-score CSV at ./data/human_score.csv (cols: img0, img1, human).
#
# Outputs
#   * out_gavel_split_{1..6}.csv         per-split GAVEL predictions
#   * report_gavel_split_{1..6}.json     per-split parse-fail / mode counts
#   * corr_summary_split_{1..6}.csv      per-split metric correlations
#   * significance_table.csv             aggregated mean ± std + Wilcoxon p

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs/eval outputs/corr outputs/reports

VLM_PATH="${VLM_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
FUSION_CKPT="${FUSION_CKPT:-./checkpoints/refclip_fusion.pt}"
DECAY="${DECAY:-super2}"

for k in 1 2 3 4 5 6; do
    PAIRS="./data/pairs_split${k}.csv"
    OUT="./outputs/eval/out_gavel_split_${k}.csv"
    REP="./outputs/reports/report_gavel_split_${k}.json"

    if [[ ! -f "${PAIRS}" ]]; then
        echo "[skip] missing ${PAIRS}"
        continue
    fi

    echo "=== split ${k} ==="
    python scripts/eval.py \
        --pairs_csv  "${PAIRS}" \
        --save_csv   "${OUT}"   \
        --report_json "${REP}"  \
        --vlm_model_path "${VLM_PATH}" \
        --fusion_ckpt   "${FUSION_CKPT}" \
        --decay_kind    "${DECAY}"
done

# ---- correlation against human scores ---------------------------------------
# If you want to recompute correlations against external baselines, point
# CORR_SCRIPT and BASELINE_DIR at your local evaluation utilities/results.

CORR_SCRIPT="${CORR_SCRIPT:-}"
HUMAN_CSV="${HUMAN_CSV:-./data/human_score.csv}"
BASELINE_DIR="${BASELINE_DIR:-./outputs/baselines}"

if [[ -f "${CORR_SCRIPT}" && -f "${HUMAN_CSV}" ]]; then
    for k in 1 2 3 4 5 6; do
        OUT="./outputs/eval/out_gavel_split_${k}.csv"
        SUM="./outputs/corr/corr_summary_split_${k}.csv"
        [[ -f "${OUT}" ]] || continue
        python "${CORR_SCRIPT}" \
            --human_csv "${HUMAN_CSV}" \
            --metric_csv \
                "${OUT}" \
                "${BASELINE_DIR}/out_lpips_${k}.csv" \
                "${BASELINE_DIR}/out_dists_${k}.csv" \
                "${BASELINE_DIR}/out_siglip_full_${k}.csv" \
                "${BASELINE_DIR}/out_openclip_full_${k}.csv" \
                "${BASELINE_DIR}/out_dreamsim_schemaA_${k}.csv" \
                "${BASELINE_DIR}/out_dinov2_full_${k}.csv" \
                "${BASELINE_DIR}/out_traditional_full_${k}.csv" \
            --save_summary "${SUM}" \
            --verbose
    done
fi

# ---- mean ± std + Wilcoxon --------------------------------------------------
shopt -s nullglob
CORR_FILES=(./outputs/corr/corr_summary_split_*.csv)
if (( ${#CORR_FILES[@]} )); then
    python scripts/significance.py \
        --summary_glob "./outputs/corr/corr_summary_split_*.csv" \
        --reference   out_gavel_split \
        --out_csv     ./outputs/significance_table.csv \
        --out_matrix_csv ./outputs/significance_matrix.csv
    echo "[done] see outputs/significance_table.csv"
else
    echo "[skip] no correlation summaries found; provide HUMAN_CSV, CORR_SCRIPT, and baseline CSVs to build the significance table"
fi
