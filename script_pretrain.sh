#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0


python -m torch.distributed.launch --nproc_per_node=1 --master_port 11905 pretrain.py \
--lr 0.0001 \
--batch-size 16 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 1024 \
--mlp \
--contrast-t 0.07 \
--contrast-k 6912 \
--checkpoint-path save_path \
--schedule 100 \
--epochs 10 \
--pre-dataset SLR \
--skeleton-representation graph-based \
--worker 8 \
--optim Adam \
--warmup \
--wd 0.03 \
--contrast_weight 0.05 \
--mask_ratio 0.9 \
