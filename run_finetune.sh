#!/bin/bash
python ec_prediction_with_finetune.py \
  --mode train \
  --num_gpus 3 \
  --epochs 100 \
  --batch_size 16 \
  --lr 5e-5 \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --gradient_accumulation_steps 4 \
  --save_path TEC_Fusion/checkpoints/ec_model_finetuned.pth