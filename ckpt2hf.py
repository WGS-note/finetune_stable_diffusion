# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/3/27 17:39
# @File: run_t2i.py
'''
text to image based on lora
'''
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora.lora_diffusion import tune_lora_scale, patch_pipe, UNET_EXTENDED_TARGET_REPLACE
from lora.lora_diffusion import image_grid, cli_lora_add
from tools.utils import convert_full_checkpoint
import torch
import fire
from safetensors import safe_open

from tools.show_grid import show_grid

import warnings, time
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ckpt2diffusers(safetensor_path, HF_MODEL_DIR, config_file):
    '''

    :param safetensor_path:
    :param HF_MODEL_DIR:
    :param vae_pt_path:
    :param config_file:
    :return:
    '''
    scheduler_type = "PNDM"  # K-LMS / DDIM / EulerAncestral / K-LMS
    sss = time.time()

    convert_full_checkpoint(
        safetensor_path,
        config_file,
        scheduler_type=scheduler_type,
        extract_ema=False,
        output_path=HF_MODEL_DIR,
        vae_pt_path=None,
        with_control_net=False
    )

    print('convert safetensor to diffusers ok --- {}'.format(time.time() - sss))

def inference(base_model='./weights/3Guofeng3_v32Light', save_img='./examples/'):

    base_model = './weights/anythingV5Anything'

    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # prompt = "best quality, masterpiece, 1girl, china dress, Beautiful face"
    negative_prompt = "(((simple background))),monochrome ,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, lowres, bad anatomy, bad hands, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mut ilated,tran nsexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,bad anatomy,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (((missing arms))),(((missing legs))), (((extra arms))),(((extra legs))),pubic hair, plump,bad legs,error legs,username,blurry,bad feet"

    prompts = ["A true portrait of George Washington",
               "Jack Ma and Musk eat",
               "belle",
               "Running girl",
               "mona lisa",
               "Wife and children",
               "A kitten with a plate and a fish in his mouth is pouting his butt and looking out the window",
               "A boy holds a kitten",
               "Fitness long-haired beauty",
               "Obama at Summoner Canyon",
               "Two-dimensional beauty",
               "beautiful woman and handsome man",
               "A nurse is with the patient",
               "Young girl, beautiful, tall, light skirt",
               "A man with a good heart",
               ]
    show_grid(prompts=prompts, pipe=pipe, save_path='./examples/15-35.jpg', hn=5, wn=3, num_inference_steps=30, width=512, height=512)

def safe_op(loar_path):
    tensors = {}
    with safe_open(loar_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # print(tensors)
    return tensors


if __name__ == '__main__':
    '''
    docker run -d --gpus '"device=1"' \
               --rm -it --name lora_diff \
               --shm-size 12G \
               -v /data/wgs/LoRA:/home \
               wgs-torch:control_diffusion \
               sh -c "python -u /home/ckpt2diffusers.py 1>>/home/log/ckpt2diffusers.log 2>>/home/log/ckpt2diffusers.err"
               
    
    √ CHECKPOINT
    '''

    # safetensors checkpoint 格式转 hf
    safetensor_path = './weights/anythingV5Anything_anythingV5PrtRE.safetensors'
    HF_MODEL_DIR = './weights/anythingV5Anything'
    config_file = "./stable-diffusion-v1-5/v1-inference.yaml"
    ckpt2diffusers(safetensor_path=safetensor_path, HF_MODEL_DIR=HF_MODEL_DIR, config_file=config_file)
    # # ''' ???
    # # $ lora_add runwayml/stable-diffusion-v1-5 ./example_loras/lora_krk.safetensors ./output_merged 0.8 --mode upl
    # # '''
    # # fire.Fire(cli_lora_add.add('./stable-diffusion-v1-5', safetensor_path, HF_MODEL_DIR, mode='upl'))
    #
    # fire.Fire(cli_lora_add.add('./stable-diffusion-v1-5',
    #                            './weights/lora_samdoesartsSamYang_offsetRightFilesize.safetensors',
    #                            './weights/lora_samdoesartsSamYang_offsetRightFilesize',
    #                            mode='upl'))

    #
    inference()




