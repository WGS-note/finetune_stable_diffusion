#!/bin/bash
# 不运行这个sh，直接docker run
# 详见：
# https://docs.wandb.ai/quickstart
# https://wangguisen.blog.csdn.net/article/details/130260950

# container name: lora_diff
docker run -it --gpus '"device=3"' --name lora_diff\
                   --shm-size 16G \
                   -v /data/wgs/finetune_stable_diffusion:/home \
                   wgs-torch:control_diffusion \
                   bash

# wandb init

