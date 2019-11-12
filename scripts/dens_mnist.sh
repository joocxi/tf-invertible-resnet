#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode train \
    --dataset mnist \
    --block_list 32,32,32 \
    --stride_list 1,2,2 \
    --channel_list 32,32,32 \
    --num_series_terms 10 \
    --learning_rate 0.003

CUDA_VISIBLE_DEVICES=0 python main.py --mode train \
    --dataset mnist \
    --block_list 1 \
    --stride_list 1 \
    --channel_list 32 \
    --num_series_terms 1 \
    --learning_rate 0.003 \
