# 133

#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
#
#RUN echo "" > /etc/apt/sources.list.d/cuda.list
#RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
#RUN sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
#RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
#RUN apt-get install -y cmake
#RUN apt-get install -y git
#
#RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy ftfy wandb safetensors mediapipe fire OmegaConf
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub==0.11.1 transformers==4.25.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers==0.15.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate==0.16.0 modelcards==0.1.6
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
#RUN apt-get install libgl1-mesa-glx -y
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentencepiece sacremoses datasets
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard Jinja2 bitsandbytes
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xformers==0.0.17.dev476
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple triton==2.0.0.dev20221120
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torchvision==0.14.1

#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade diffusers
#WORKDIR /home
#
## --upgrade
## docker build -t wgs-torch:control_diffusion -f ./dk/Dockerfile .



# ------------------------------
FROM wgs-torch:control_diffusion
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers==0.15.1
WORKDIR /home




