#!/usr/bin/env bash

set -eu

# $1 生成数据的scp文件地址
# $2 clean.scp地址
./nnet/compute_si_snr.py $1 $2

# 运行样例
# ./test.sh ./sps_tas/spk1.scp ./data/tt/spk1.scp