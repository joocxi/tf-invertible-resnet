#!/usr/bin/env bash
python main.py --mode train \
    --dataset cifar10 \
    --block_list 32 32 32 \
    --stride_list 1 2 2 \
    --channel_list 512 512 512 \
    --num_series_terms 5 \
    --learning_rate 0.003
