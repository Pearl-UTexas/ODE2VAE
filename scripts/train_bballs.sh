#!/bin/bash
python train.py \
  --task bballs \
  --data_root "data/bouncing_balls_data/noise_0.1" \
  --q 25 \
  --Hf 100 \
  --amort_len 3 \
  --batch_size 10 \
  --activation_fn "relu" \
  --eta 0.0008
