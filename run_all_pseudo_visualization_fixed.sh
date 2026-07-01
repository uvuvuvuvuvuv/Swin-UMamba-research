#!/usr/bin/env bash
set -euo pipefail

source /home/baiyuting/anaconda3/etc/profile.d/conda.sh
conda activate swin_umamba

cd /storage/baiyuting/data/Swin-UMamba-main

PROCESSED_ROOT=/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed
VIS_ROOT=/storage/baiyuting/data/out_data_idea1/visualization/pseudo_quality_fixed
DATASETS=kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,ph2,acdc,prostate158,btcv,synapse

echo "=================================================="
echo "[1/2] Full + Box：native_full"
echo "=================================================="

OUT_FULL="$VIS_ROOT/all_native_full"

rm -rf "$OUT_FULL"
mkdir -p "$OUT_FULL"

python -u tools/visualize_pseudo_quality.py \
  --processed_root "$PROCESSED_ROOT" \
  --output_root "$OUT_FULL" \
  --datasets "$DATASETS" \
  --fold fold_0 \
  --method idea1_sac_medsam_final \
  --baseline_pseudo_name tri_train \
  --sac_pseudo_name tri_train_idea1_sac_medsam_final \
  --prompt_space teacher \
  --display_mode native_full \
  --grayscale_datasets tn3k,tg3k,ddti,otu_2d \
  --per_group 12 \
  --max_per_case 3 \
  --overlay_alpha 0.58 \
  --error_alpha 0.62 \
  --dpi 220 \
  --sheet_dpi 180 \
  --include_full \
  --no-render_individual \
  --render_contact_sheets \
  --strict

echo
echo "=================================================="
echo "[2/2] Box-only：native_focus"
echo "=================================================="

OUT_FOCUS="$VIS_ROOT/all_native_focus"

rm -rf "$OUT_FOCUS"
mkdir -p "$OUT_FOCUS"

python -u tools/visualize_pseudo_quality.py \
  --processed_root "$PROCESSED_ROOT" \
  --output_root "$OUT_FOCUS" \
  --datasets "$DATASETS" \
  --fold fold_0 \
  --method idea1_sac_medsam_final \
  --baseline_pseudo_name tri_train \
  --sac_pseudo_name tri_train_idea1_sac_medsam_final \
  --prompt_space teacher \
  --display_mode native_focus \
  --focus_margin 0.18 \
  --grayscale_datasets tn3k,tg3k,ddti,otu_2d \
  --per_group 12 \
  --max_per_case 3 \
  --overlay_alpha 0.58 \
  --error_alpha 0.62 \
  --dpi 220 \
  --sheet_dpi 180 \
  --only_box \
  --render_individual \
  --render_contact_sheets \
  --strict

echo
echo "=================================================="
echo "[DONE] 全部可视化生成完成"
echo "完整视野：$OUT_FULL"
echo "聚焦诊断：$OUT_FOCUS"
echo "=================================================="
