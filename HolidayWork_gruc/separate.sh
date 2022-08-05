#!/usr/bin/env bash

set -eu

# $1 模型地址
# $2 测试集mix.scp地址
./nnet/separate.py $1 --input $2 --gpu 0 --fs 16000 > separate.log 2>&1

# 运行样例
# ./separate.sh ./exp/3 ./data/tt/mix.scp