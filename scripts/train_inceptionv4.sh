#!/usr/bin/env bash

python train.py \
    --model inceptionv4 \
    --pretrained \
    --image_size 299 \
    --batch_size 32 \
    --lr 0.0045 \
    --print-freq 50 \
    --tensorboard \
    --gpu_ids 0,1,2,3 \
    --expname 11-fold-1 \
    --data_root data/training/data/ \
    --train_file_list train_data_lists/11-fold-1-train.csv \
    --val_file_list train_data_lists/11-fold-1-val.csv
    