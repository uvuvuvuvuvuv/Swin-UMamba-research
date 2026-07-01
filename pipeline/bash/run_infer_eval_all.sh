#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

MEDROOT=/storage/baiyuting/data/MedSAM-main
SWINROOT=/storage/baiyuting/data/Swin-UMamba-main
PROC=$MEDROOT/data/processed
WORK=$SWINROOT/work_dir

DATASETS_3D=("btcv" "synapse" "acdc" "prostate158")
DATASETS_2D=("kvasirseg" "cvc_clinicdb" "tn3k" "tg3k" "ddti" "otu_2d" "monuseg" "ph2")
DATASETS_FILTER_STR=${DATASETS_FILTER:-}

BASELINE_GPU_QUEUE_STR=${BASELINE_GPU_QUEUE:-2}
UPPER_GPU_QUEUE_STR=${UPPER_GPU_QUEUE:-3}

need_file () {
  if [ ! -e "$1" ]; then
    echo "[ERROR] missing: $1" >&2
    exit 1
  fi
}

need_file "$SWINROOT/pipeline/infer_student.py"
need_file "$SWINROOT/pipeline/eval_2d.py"
need_file "$SWINROOT/pipeline/eval_3d.py"

infer_one () {
  GPU_ID=$1
  DS=$2
  MODE=$3

  CKPT=$WORK/$MODE/$DS/fold_0/last.pth
  OUTDIR=$WORK/$MODE/$DS/fold_0/pred_test

  need_file "$CKPT"

  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"

  echo
  echo "[$(date '+%F %T')] INFER $MODE / $DS on GPU $GPU_ID"

  CUDA_VISIBLE_DEVICES=$GPU_ID python -u $SWINROOT/pipeline/infer_student.py \
    --fold_root $PROC/$DS/fold_0 \
    --dataset $DS \
    --mode $MODE \
    --deep_supervision \
    --ckpt $CKPT \
    --out_dir $OUTDIR \
    --amp \
    2>&1 | tee $WORK/$MODE/$DS/fold_0/run_infer_${MODE}.log
}

eval_one_3d () {
  DS=$1
  MODE=$2

  rm -rf $WORK/$MODE/$DS/fold_0/eval_3d
  mkdir -p $WORK/$MODE/$DS/fold_0/eval_3d

  echo
  echo "[$(date '+%F %T')] EVAL3D $MODE / $DS"

  python -u $SWINROOT/pipeline/eval_3d.py \
    --fold_root $PROC/$DS/fold_0 \
    --pred_dir $WORK/$MODE/$DS/fold_0/pred_test \
    --save_dir $WORK/$MODE/$DS/fold_0/eval_3d \
    --split test \
    --require_native_gt \
    2>&1 | tee $WORK/$MODE/$DS/fold_0/run_eval_${MODE}.log
}

eval_one_2d () {
  DS=$1
  MODE=$2

  EXTRA=""
  if [ "$DS" = "kvasirseg" ] || [ "$DS" = "cvc_clinicdb" ]; then
    EXTRA="--report_mae"
  fi

  rm -rf $WORK/$MODE/$DS/fold_0/eval_2d
  mkdir -p $WORK/$MODE/$DS/fold_0/eval_2d

  echo
  echo "[$(date '+%F %T')] EVAL2D $MODE / $DS"

  python -u $SWINROOT/pipeline/eval_2d.py \
    --fold_root $PROC/$DS/fold_0 \
    --pred_dir $WORK/$MODE/$DS/fold_0/pred_test \
    --save_dir $WORK/$MODE/$DS/fold_0/eval_2d \
    --split test \
    --require_native_gt \
    $EXTRA \
    2>&1 | tee $WORK/$MODE/$DS/fold_0/run_eval_${MODE}.log
}

split_csv_to_array () {
  local csv="$1"
  local -n arr_ref=$2
  IFS=',' read -r -a arr_ref <<< "$csv"
}

dataset_selected () {
  local ds=$1
  if [ -z "$DATASETS_FILTER_STR" ]; then
    return 0
  fi
  local items=()
  split_csv_to_array "$DATASETS_FILTER_STR" items
  local item
  for item in "${items[@]}"; do
    if [ "$item" = "$ds" ]; then
      return 0
    fi
  done
  return 1
}

wait_for_slot () {
  local -n pid_ref=$1
  local -n gpu_ref=$2
  local count=${#pid_ref[@]}
  local idx
  while true; do
    for ((idx=0; idx<count; idx++)); do
      local pid="${pid_ref[$idx]}"
      if [ -z "$pid" ]; then
        echo "$idx"
        return 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || return 1
        pid_ref[$idx]=""
        echo "$idx"
        return 0
      fi
    done
    sleep 5
  done
}

launch_infer_queue () {
  local mode=$1
  local queue_csv=$2
  shift 2
  local datasets=("$@")

  local gpus=()
  split_csv_to_array "$queue_csv" gpus
  local pids=()
  local idx ds slot gpu
  for ((idx=0; idx<${#gpus[@]}; idx++)); do
    pids+=("")
  done

  echo "[CONFIG] infer mode=$mode gpu_queue=${gpus[*]}"

  for ds in "${datasets[@]}"; do
    slot=$(wait_for_slot pids gpus) || return 1
    gpu=${gpus[$slot]}
    infer_one "$gpu" "$ds" "$mode" &
    pids[$slot]=$!
    sleep 2
  done

  for pid in "${pids[@]}"; do
    if [ -n "$pid" ]; then
      wait "$pid"
    fi
  done
}

ALL_DATASETS_3D=()
ALL_DATASETS_2D=()
for ds in "${DATASETS_3D[@]}"; do
  if dataset_selected "$ds"; then
    ALL_DATASETS_3D+=("$ds")
  fi
done
for ds in "${DATASETS_2D[@]}"; do
  if dataset_selected "$ds"; then
    ALL_DATASETS_2D+=("$ds")
  fi
done

echo "[CONFIG] BASELINE_GPU_QUEUE=$BASELINE_GPU_QUEUE_STR"
echo "[CONFIG] UPPER_GPU_QUEUE=$UPPER_GPU_QUEUE_STR"
echo "[CONFIG] DATASETS_FILTER=${DATASETS_FILTER_STR:-<all>}"
echo "[CONFIG] pred dir = work_dir/<mode>/<dataset>/fold_0/pred_test"
echo "[CONFIG] eval dir = work_dir/<mode>/<dataset>/fold_0/eval_2d or eval_3d"

launch_infer_queue baseline "$BASELINE_GPU_QUEUE_STR" "${ALL_DATASETS_3D[@]}" "${ALL_DATASETS_2D[@]}" &
PID_BASELINE=$!
sleep 2
launch_infer_queue upper "$UPPER_GPU_QUEUE_STR" "${ALL_DATASETS_3D[@]}" "${ALL_DATASETS_2D[@]}" &
PID_UPPER=$!

wait $PID_BASELINE
wait $PID_UPPER

for ds in "${DATASETS_3D[@]}"; do
  eval_one_3d $ds baseline
  eval_one_3d $ds upper
done

for ds in "${DATASETS_2D[@]}"; do
  eval_one_2d $ds baseline
  eval_one_2d $ds upper
done
