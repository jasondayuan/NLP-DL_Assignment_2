#!/bin/bash

model="$1"

experiments=(1 2 3 4 5)

for exp in "${experiments[@]}"; do
  CUDA_VISIBLE_DEVICES=5 proxychains4 python train.py \
  --model_name_or_path "$1" \
  --dropout 0.3 \
  --exp "$exp"\
  --dataset_name agnews_sup \
  --output_dir ./results/"$1"-agnews_sup \
  --num_train_epochs 35 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 9e-5 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --logging_dir ./logs \
  --report_to wandb \
  --weight_decay 0.01 \
  --warmup_steps 70 \
  --seed "$exp"
done