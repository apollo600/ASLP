#!/usr/bin/env bash

set -eu

epochs=50
batch_size=16
cpt_dir=exp

#echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
./nnet/train.py \
  --gpu 0 \
  --epochs $epochs \
  --checkpoint $cpt_dir/$1 \
  --batch-size $batch_size \
  > $1.train.log 2>&1

# 运行样例
# ./train.sh 3

# --resume ./exp/2/best.pt.tar \
