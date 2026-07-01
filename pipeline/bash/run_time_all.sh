#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

MEDROOT=/storage/baiyuting/data/MedSAM-main
SWINROOT=/storage/baiyuting/data/Swin-UMamba-main
PROC=$MEDROOT/data/processed
WORK=$SWINROOT/work_dir

DATASETS="btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,monuseg,ph2"

cd $SWINROOT/pipeline

python -u timer.py \
  --processed_root $PROC \
  --work_dir $WORK \
  --datasets $DATASETS \
  --save_json $WORK/pipeline_time_all.json \
  --save_csv $WORK/pipeline_time_all.csv \
  --save_paper_csv $WORK/pipeline_time_all_paper.csv \
  --save_paper_md $WORK/pipeline_time_all_paper.md