#!/bin/bash
# Usage:
# ./scripts/eval.sh GPU MODEL_PATH IMG_NUM
#
# Example:
# ./scripts/eval.sh 0 output/res101_trained_weights/v1_shelf/shelf_51model.ckpt 20

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
MODEL_PATH=$2
IMG_NUM=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

time CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval.py \
     --weights ${MODEL_PATH} \
     --img_num ${IMG_NUM} \
     ${EXTRA_ARGS}
