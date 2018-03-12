#!/bin/bash
# Usage:
# ./scripts/inference.sh GPU MODEL_PATH
#
# Example:
# ./scripts/inference.sh 0 output/res101_trained_weights/v1_shelf/shelf_51model.ckpt

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
MODEL_PATH=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

time CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/inference.py \
     --weights ${MODEL_PATH} \
     ${EXTRA_ARGS}
