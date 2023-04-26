#!/bin/bash

cd /home

accelerate config default

export MODEL_NAME="./stable-diffusion-v1-5"
#export OUTPUT_DIR="./weights/sd1-5_lora"
#export TRAIN_DATA_DIR="./data/lora_test_data"
export OUTPUT_DIR="./weights/sd1-5_lora2"
export TRAIN_DATA_DIR="./data/lora_test_data2"

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR --caption_column="text" \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=3000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=1337 \
  --validation_prompt="summer palace in spring" \
  --num_validation_images=4 \
  --checkpointing_steps=30000

# --train_batch_size=3

# --max_train_steps=15000 \
# --num_train_epochs=10000
#--validation_prompt=""
#--caption_column="text"      "image"
#--num_train_epochs=100
#use_8bit_adam
#enable_xformers_memory_efficient_attention
#--dataset_name=$DATASET_NAME --caption_column="text" \
#--dataloader_num_workers=0 \

#--report_to=wandb \  日日志集成位置，默认为 tensorboard，如果为 wandb，详见 init_wb.sh

# --multi_gpu
