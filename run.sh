#!/bin/bash
# training command for different datasets.

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --model "dagcn" \
  --type "train" \
	--dataset "twitter" \
	--vocab_dir "./dataset/Twitter" \
	--glove_dir "./glove/glove.840B.300d.txt" \
	--output_merge "biaffine" \
	--syn_layers 2 \
	--sem_layers 1 \
	--sem_att_heads 5 \
	--num_epoch 60 \
	--lr 0.002 \
	--l2reg 0.0001 \
