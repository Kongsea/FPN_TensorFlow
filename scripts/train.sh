#!/bin/bash
# Usage:
# ./scripts/train.sh GPU
#
# Example:
# ./scripts/train.sh 0 coca

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

# mkdir "logs"
LOG="logs/FPN_${DATASET}.txt.`date +'%Y_%m_%d_%H_%M_%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py ${EXTRA_ARGS}
