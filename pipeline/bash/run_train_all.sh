#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

MEDROOT=/storage/baiyuting/data/MedSAM-main
SWINROOT=/storage/baiyuting/data/Swin-UMamba-main
PROC=$MEDROOT/data/processed
WORK=$SWINROOT/work_dir
CKPT=$SWINROOT/data/pretrained/vmamba/vmamba_tiny_e292.pth

DATASETS_3D=("btcv" "synapse" "acdc" "prostate158")
DATASETS_2D=("kvasirseg" "cvc_clinicdb" "tn3k" "tg3k" "ddti" "otu_2d" "monuseg" "ph2")
DATASETS_FILTER_STR=${DATASETS_FILTER:-}
MODE_FILTER=${MODE_FILTER:-both}

# 默认保持当前稳定配置；如需扩展并行，可通过环境变量覆盖：
#   BASELINE_GPU_QUEUE=0,2
#   UPPER_GPU_QUEUE=3
# 或者：
#   BASELINE_GPU_QUEUE=0,2
#   UPPER_GPU_QUEUE=1,3
BASELINE_GPU_QUEUE_STR=${BASELINE_GPU_QUEUE:-2}
UPPER_GPU_QUEUE_STR=${UPPER_GPU_QUEUE:-3}

need_file () {
  if [ ! -e "$1" ]; then
    echo "[ERROR] missing: $1" >&2
    exit 1
  fi
}

need_file "$CKPT"
need_file "$SWINROOT/pipeline/train_student.py"

dataset_cfg () {
  DS=$1
  case "$DS" in
    btcv)         echo "1 0" ;;   # bs nw
    synapse)      echo "1 0" ;;
    acdc)         echo "2 2" ;;
    prostate158)  echo "2 2" ;;
    kvasirseg)    echo "8 4" ;;
    cvc_clinicdb) echo "8 4" ;;
    tn3k)         echo "8 4" ;;
    tg3k)         echo "8 4" ;;
    ddti)         echo "8 4" ;;
    otu_2d)       echo "8 4" ;;
    monuseg)      echo "8 4" ;;
    ph2)          echo "8 4" ;;
    *)            echo "2 4" ;;
  esac
}

run_one () {
  GPU_ID=$1
  DS=$2
  MODE=$3
  BS=$4
  NW=$5

  OUTDIR=$WORK/$MODE/$DS/fold_0

  # 覆盖重写：保持原目录结构，不新建分叉目录
  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"

  echo
  echo "[$(date '+%F %T')] TRAIN $MODE / $DS on GPU $GPU_ID | bs=$BS nw=$NW"

  CUDA_VISIBLE_DEVICES=$GPU_ID python -u $SWINROOT/pipeline/train_student.py \
    --fold_root $PROC/$DS/fold_0 \
    --dataset $DS \
    --mode $MODE \
    --epochs 50 \
    --batch_size $BS \
    --num_workers $NW \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --freeze_encoder_epochs 10 \
    --amp \
    --deep_supervision \
    --pretrained_ckpt $CKPT \
    --out_dir $OUTDIR \
    2>&1 | tee $OUTDIR/run_train_${MODE}.log
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

launch_mode_queue () {
  local mode=$1
  local queue_csv=$2
  shift 2
  local datasets=("$@")

  local gpus=()
  split_csv_to_array "$queue_csv" gpus
  local pids=()
  local idx ds bs nw slot gpu
  for ((idx=0; idx<${#gpus[@]}; idx++)); do
    pids+=("")
  done

  echo "[CONFIG] mode=$mode gpu_queue=${gpus[*]}"

  for ds in "${datasets[@]}"; do
    read bs nw <<< "$(dataset_cfg $ds)"
    slot=$(wait_for_slot pids gpus) || return 1
    gpu=${gpus[$slot]}
    run_one "$gpu" "$ds" "$mode" "$bs" "$nw" &
    pids[$slot]=$!
    sleep 2
  done

  for pid in "${pids[@]}"; do
    if [ -n "$pid" ]; then
      wait "$pid"
    fi
  done
}

ALL_DATASETS=()
for ds in "${DATASETS_3D[@]}"; do
  if dataset_selected "$ds"; then
    ALL_DATASETS+=("$ds")
  fi
done
for ds in "${DATASETS_2D[@]}"; do
  if dataset_selected "$ds"; then
    ALL_DATASETS+=("$ds")
  fi
done

echo "[CONFIG] BASELINE_GPU_QUEUE=$BASELINE_GPU_QUEUE_STR"
echo "[CONFIG] UPPER_GPU_QUEUE=$UPPER_GPU_QUEUE_STR"
echo "[CONFIG] DATASETS_FILTER=${DATASETS_FILTER_STR:-<all>}"
echo "[CONFIG] MODE_FILTER=$MODE_FILTER"
echo "[CONFIG] DATASETS_3D=${DATASETS_3D[*]}"
echo "[CONFIG] DATASETS_2D=${DATASETS_2D[*]}"
echo "[CONFIG] btcv/synapse fixed as bs=1 nw=0"

PID_BASELINE=""
PID_UPPER=""

if [ "$MODE_FILTER" = "baseline" ] || [ "$MODE_FILTER" = "both" ]; then
  launch_mode_queue baseline "$BASELINE_GPU_QUEUE_STR" "${ALL_DATASETS[@]}" &
  PID_BASELINE=$!
  sleep 2
fi

if [ "$MODE_FILTER" = "upper" ] || [ "$MODE_FILTER" = "both" ]; then
  launch_mode_queue upper "$UPPER_GPU_QUEUE_STR" "${ALL_DATASETS[@]}" &
  PID_UPPER=$!
fi

if [ -n "$PID_BASELINE" ]; then
  wait $PID_BASELINE
fi
if [ -n "$PID_UPPER" ]; then
  wait $PID_UPPER
fi
