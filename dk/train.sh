#!/bin/bash

path="/data/wgs/finetune_stable_diffusion"
cd $path

if [ "${1}" = "dreambooth" ]; then
  docker run --rm -it -d --gpus '"device=3"' --name dreambooth_diff\
                   --shm-size 16G \
                   -v /data/wgs/finetune_stable_diffusion:/home \
                   wgs-torch:control_diffusion \
                   sh -c "sh /home/dk/train_dreambooth.sh 1>>/home/log/train_dreambooth.log 2>>/home/log/train_dreambooth.err"
elif [ "${1}" = "lora" ]; then
  docker run --rm -it -d --gpus '"device=3"' --name lora_diff\
                   --shm-size 16G \
                   -v /data/wgs/finetune_stable_diffusion:/home \
                   wgs-torch:control_diffusion \
                   sh -c "sh /home/dk/train_lora.sh 1>>/home/log/train_lora.log 2>>/home/log/train_lora.err"
elif [ "${1}" = "dreambooth_lora" ]; then
  echo "333"
elif [ "${1}" = "controlnet" ]; then
  echo "333"
else
  echo "error"
fi


# sh ./dk/train.sh dreambooth
# sh ./dk/train.sh lora
# sh ./dk/train.sh dreambooth_lora
# sh ./dk/train.sh controlnet