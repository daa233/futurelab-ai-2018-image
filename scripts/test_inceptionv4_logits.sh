#!/usr/bin/env bash

python test.py \
    --model inceptionv4 \
    --pretrained \
    --image_size 299 \
    --batch_size 48 \
    --print-freq 50 \
    --gpu_ids 0,1,2 \
    --expname 11-fold-1 \
    --test_data_root /media/ouc/4T_B/DuAngAng/datasets/futurelab/test/image_scene_test_b_0515/data/ \
    --test_file_list test_data_lists/list.csv \
    --resume checkpoints/inceptionv4_11-fold-1/model_best.pth.tar \
    --save_logits


