#!/bin/bash

cd /home

accelerate config default

export MODEL_NAME="./stable-diffusion-v1-5"
export INSTANCE_DIR="./data/keji"
export CLASS_DIR="./data/dog_class"
export OUTPUT_DIR="./weights/sd1-5_dreambooth"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="kejisks dog" \
  --class_prompt="dog" \
  --revision="float16" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention

# --instance_prompt="a photo of sks dog" \
# --class_prompt="a photo of dog" \
# --revision="float16" \ float32
# --mixed_precision="fp16" \
#  --set_grads_to_none \
# --checkpointing_steps=500 \

#--MODEL_NAME：base model
#--INSTANCE_DIR：微调数据集
#--CLASS_DIR：用以先验损失的图像，通常为 num_epochs * num_samples ，num_class_images设置生成的数量，通常200-300
#--OUTPUT_DIR：模型输出路径
#--with_prior_preservation，--prior_loss_weight=1.0：分别是使用先验知识保留和先验损失权重
## 如果你的数据样本比较少，那么可以使用这两个参数，可以提升训练效果，还可以防止过拟合（即生成的图片与训练的图片相似度过高）
#--instance_prompt：微调注入新的概念prompt（注意，e.g.：kejisks dog）
#--class_prompt：先验prompt，使用同一类的其他图像作为训练过程的一部分（注意，e.g.：dog）
#--revision：精度
#--mixed_precision：混合精度
#--resolution：input feature map size
#--gradient_accumulation_steps：梯度积累步骤
#----lr_scheduler：可选项有constant, linear, cosine, cosine_with_restarts, cosine_with_hard_restarts
## 学习率调整策略，一般是constant，即不调整
#--lr_warmup_steps，如果你使用的是constant，那么这个参数可以忽略，
## 如果使用其他的，那么这个参数可以设置为0，即不使用warmup
## 也可以设置为其他的值，比如1000，即在前1000个step中，学习率从0慢慢增加到learning_rate的值
## 一般不需要设置, 除非你的数据集很大，训练收敛很慢
#--num_class_images：调用模型生成图像，提示词为 class_prompt，保存位置为 CLASS_DIR
#--max_train_steps：训练的最大步数 max_train_steps  e.g.: len(imgs) * 100
#--checkpointing_steps：多少steps保存ckpt
#--use_8bit_adam：使用8bit
#--enable_xformers_memory_efficient_attention：使用xformers