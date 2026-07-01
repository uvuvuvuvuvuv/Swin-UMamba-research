#!/usr/bin/env bash
set -uo pipefail

SWIN_ROOT="/storage/baiyuting/data/Swin-UMamba-main"
MEDSAM_ROOT="/storage/baiyuting/data/MedSAM-main"
PRETRAINED_CKPT="${SWIN_ROOT}/data/pretrained/vmamba/vmamba_tiny_e292.pth"

DATASETS=(btcv synapse acdc prostate158 kvasirseg cvc_clinicdb tn3k tg3k ddti otu_2d ph2)
GPUS=(0 1 2 3)

get_bs_nw() {
  local d="$1"
  case "$d" in
    btcv|synapse)
      echo "1 0"
      ;;
    acdc|prostate158)
      echo "2 2"
      ;;
    *)
      echo "8 4"
      ;;
  esac
}

run_one_train() {
  local gpu="$1"
  local dataset="$2"
  local mode="$3"

  read -r bs nw < <(get_bs_nw "$dataset")

  local fold_root="${MEDSAM_ROOT}/data/processed/${dataset}/fold_0"
  local out_dir="${SWIN_ROOT}/work_dir/${mode}/${dataset}/fold_0"
  local log_file="${out_dir}/run_train_${mode}.log"

  if [[ ! -d "$fold_root" ]]; then
    echo "[ERROR] fold_root not found: $fold_root"
    return 1
  fi

  if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "[ERROR] pretrained checkpoint not found: $PRETRAINED_CKPT"
    return 1
  fi

  echo "============================================================"
  echo "[TRAIN] dataset=${dataset} mode=${mode} gpu=${gpu} bs=${bs} nw=${nw}"
  echo "============================================================"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  {
    echo "[START] $(date '+%F %T')"
    echo "dataset=${dataset}"
    echo "mode=${mode}"
    echo "gpu=${gpu}"
    echo "batch_size=${bs}"
    echo "num_workers=${nw}"
    echo "fold_root=${fold_root}"
    echo "out_dir=${out_dir}"
    echo "pretrained_ckpt=${PRETRAINED_CKPT}"
    echo
  } | tee "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" python -u "${SWIN_ROOT}/pipeline/train_student.py" \
    --fold_root "$fold_root" \
    --dataset "$dataset" \
    --mode "$mode" \
    --epochs 50 \
    --batch_size "$bs" \
    --num_workers "$nw" \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --freeze_encoder_epochs 10 \
    --amp \
    --deep_supervision \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --out_dir "$out_dir" \
    2>&1 | tee -a "$log_file"

  local status=${PIPESTATUS[0]}

  {
    echo
    echo "[END] $(date '+%F %T')"
    echo "[STATUS] ${status}"
  } | tee -a "$log_file"

  return "$status"
}

JOBS=()
for d in "${DATASETS[@]}"; do
  JOBS+=("${d}:baseline")
  JOBS+=("${d}:upper")
done

worker() {
  local gpu="$1"
  local worker_id="$2"
  local n_gpu="${#GPUS[@]}"

  for ((i=worker_id; i<${#JOBS[@]}; i+=n_gpu)); do
    IFS=":" read -r dataset mode <<< "${JOBS[$i]}"
    run_one_train "$gpu" "$dataset" "$mode" || return 1
  done
}

pids=()
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  worker "$gpu" "$idx" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

if [[ "$failed" -ne 0 ]]; then
  echo "[FAILED] Some training jobs failed."
  exit 1
fi

echo "[DONE] All training jobs finished."
