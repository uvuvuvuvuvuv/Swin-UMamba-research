#!/usr/bin/env bash
set -uo pipefail

SWIN_ROOT="/storage/baiyuting/data/Swin-UMamba-main"
MEDSAM_ROOT="/storage/baiyuting/data/MedSAM-main"

DATASETS=(btcv synapse acdc prostate158 kvasirseg cvc_clinicdb tn3k tg3k ddti otu_2d ph2)
GPUS=(0 1 2 3)

is_2d_dataset() {
  local d="$1"
  case "$d" in
    kvasirseg|cvc_clinicdb|tn3k|tg3k|ddti|otu_2d|ph2)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

need_mae() {
  local d="$1"
  case "$d" in
    kvasirseg|cvc_clinicdb)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_one_infer_eval() {
  local gpu="$1"
  local dataset="$2"
  local mode="$3"

  local fold_root="${MEDSAM_ROOT}/data/processed/${dataset}/fold_0"
  local out_root="${SWIN_ROOT}/work_dir/${mode}/${dataset}/fold_0"
  local ckpt="${out_root}/last.pth"
  local pred_dir="${out_root}/pred_test"

  local infer_log="${out_root}/run_infer_${mode}.log"
  local eval_log="${out_root}/run_eval_${mode}.log"

  if [[ ! -d "$fold_root" ]]; then
    echo "[ERROR] fold_root not found: $fold_root"
    return 1
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] checkpoint not found: $ckpt"
    return 1
  fi

  echo "============================================================"
  echo "[INFER+EVAL] dataset=${dataset} mode=${mode} gpu=${gpu}"
  echo "============================================================"

  rm -rf "$pred_dir"
  mkdir -p "$pred_dir"

  {
    echo "[START INFER] $(date '+%F %T')"
    echo "dataset=${dataset}"
    echo "mode=${mode}"
    echo "gpu=${gpu}"
    echo "fold_root=${fold_root}"
    echo "ckpt=${ckpt}"
    echo "pred_dir=${pred_dir}"
    echo
  } | tee "$infer_log"

  CUDA_VISIBLE_DEVICES="$gpu" python -u "${SWIN_ROOT}/pipeline/infer_student.py" \
    --fold_root "$fold_root" \
    --dataset "$dataset" \
    --mode "$mode" \
    --deep_supervision \
    --ckpt "$ckpt" \
    --out_dir "$pred_dir" \
    --amp \
    2>&1 | tee -a "$infer_log"

  local infer_status=${PIPESTATUS[0]}
  if [[ "$infer_status" -ne 0 ]]; then
    echo "[ERROR] infer failed: dataset=${dataset} mode=${mode}" | tee -a "$infer_log"
    return "$infer_status"
  fi

  {
    echo
    echo "[END INFER] $(date '+%F %T')"
    echo "[STATUS] ${infer_status}"
  } | tee -a "$infer_log"

  if is_2d_dataset "$dataset"; then
    local eval_dir="${out_root}/eval_2d"
    rm -rf "$eval_dir"
    mkdir -p "$eval_dir"

    {
      echo "[START EVAL_2D] $(date '+%F %T')"
      echo "dataset=${dataset}"
      echo "mode=${mode}"
      echo "pred_dir=${pred_dir}"
      echo "save_dir=${eval_dir}"
      echo
    } | tee "$eval_log"

    if need_mae "$dataset"; then
      CUDA_VISIBLE_DEVICES="$gpu" python -u "${SWIN_ROOT}/pipeline/eval_2d.py" \
        --fold_root "$fold_root" \
        --pred_dir "$pred_dir" \
        --save_dir "$eval_dir" \
        --split test \
        --require_native_gt \
        --report_mae \
        2>&1 | tee -a "$eval_log"
    else
      CUDA_VISIBLE_DEVICES="$gpu" python -u "${SWIN_ROOT}/pipeline/eval_2d.py" \
        --fold_root "$fold_root" \
        --pred_dir "$pred_dir" \
        --save_dir "$eval_dir" \
        --split test \
        --require_native_gt \
        2>&1 | tee -a "$eval_log"
    fi

    local eval_status=${PIPESTATUS[0]}
  else
    local eval_dir="${out_root}/eval_3d"
    rm -rf "$eval_dir"
    mkdir -p "$eval_dir"

    {
      echo "[START EVAL_3D] $(date '+%F %T')"
      echo "dataset=${dataset}"
      echo "mode=${mode}"
      echo "pred_dir=${pred_dir}"
      echo "save_dir=${eval_dir}"
      echo
    } | tee "$eval_log"

    CUDA_VISIBLE_DEVICES="$gpu" python -u "${SWIN_ROOT}/pipeline/eval_3d.py" \
      --fold_root "$fold_root" \
      --pred_dir "$pred_dir" \
      --save_dir "$eval_dir" \
      --split test \
      --require_native_gt \
      2>&1 | tee -a "$eval_log"

    local eval_status=${PIPESTATUS[0]}
  fi

  {
    echo
    echo "[END EVAL] $(date '+%F %T')"
    echo "[STATUS] ${eval_status}"
  } | tee -a "$eval_log"

  return "$eval_status"
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
    run_one_infer_eval "$gpu" "$dataset" "$mode" || return 1
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
  echo "[FAILED] Some infer/eval jobs failed."
  exit 1
fi

echo "[DONE] All infer/eval jobs finished."
