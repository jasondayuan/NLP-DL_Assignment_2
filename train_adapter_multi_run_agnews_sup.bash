#!/bin/bash

model="$1"

experiments=(1 2 3 4 5)

for exp in "${experiments[@]}"; do
  CUDA_VISIBLE_DEVICES=1 proxychains4 python train_adapter.py \
  --model_name_or_path "$1" \
  --dropout 0.3 \
  --exp "$exp"\
  --dataset_name agnews_sup \
  --output_dir ./results/adapter-"$1"-agnews_sup \
  --num_train_epochs 35 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 64 \
  --learning_rate 1e-4 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --logging_dir ./logs \
  --report_to wandb \
  --weight_decay 0.01 \
  --warmup_steps 70 \
  --train_adapter \
  --seed "$exp"
done