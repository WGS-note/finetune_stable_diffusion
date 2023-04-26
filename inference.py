# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/4/6 15:15
# @File: inference.py
'''

'''
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

from tools.show_grid import show_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(prompt, negative_prompt=None, save_img='./examples/'):

    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=7.5, height=512, width=512, ).images[0]

    image.save(save_img)

if __name__ == '__main__':
    '''
    docker run --rm -it --gpus '"device=3"' --name tran_diff\
                   -v /data/wgs/finetune_stable_diffusion:/home \
                   wgs-torch:control_diffusion \
                   bash
    
    train_dreambooth:
    sh ./dk/train.sh dreambooth
    
    train_dreambooth:
    sh ./dk/train.sh lora
    
    inference:
    docker run --rm -it -d --gpus '"device=3"' --name inference_diff\
                   -v /data/wgs/finetune_stable_diffusion:/home \
                   wgs-torch:control_diffusion \
                   sh -c "python -u /home/inference.py 1>>/home/log/inference.log 2>>/home/log/inference.err"          
    '''

    # torch.manual_seed(-1)

    base_model = './stable-diffusion-v1-5/'
    # base_model = './weights/sd1-5_dreambooth/'
    # lora_path = './weights/sd1-5_lora/'
    lora_path = './weights/sd1-5_lora2/'

    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.unet.load_attn_procs(lora_path)

    # inference(prompt='The Great Wall', save_img='./examples/1.png')
    # exit()

    # prompts = ["Yang Mi", "Yang Mi", "Yang Mi", "Yang Mi", "Yang Mi",
    #            "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring",
    #            "A good-looking girl in the virtual world", "A good-looking girl in the virtual world", "A good-looking girl in the virtual world", "A good-looking girl in the virtual world", "A good-looking girl in the virtual world",
    #            "Zibo Barbecue", "Zibo Barbecue", "Zibo Barbecue", "Zibo Barbecue", "Zibo Barbecue",
    #            ]
    prompts = ["summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring",
               "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring",
               "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring",
               "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring", "summer palace in spring",
               ]

    # prompts = ['heavy traffic on the street',
    #            "Cute squirrel",
    #            "Church in the snow",
    #            "Ancient style moon lantern flower tree",
    #            "Chicken in the woods",
    #            "The worker under the umbrella"]

    # prefix = "high resolution, 4k, best quality, high quality, "
    suffix = ", high resolution, 4k, best quality, high quality, photo realistic, realistic shadows, masterpiece, extremely detailed, sharp focus, 8k, extremely detailed wallpaper"
    for i in range(len(prompts)):
        # prompts[i] = prefix + prompts[i] + suffix
        prompts[i] = prompts[i] + suffix

    negative_prompt = 'NSFW, photography, blurry, artifacts, duplicate, mutilated, deformed, ugly, blurry, bad anatomy, lowres, bad anatomy, text, error, cropped, worst quality, low quality, normal quality, jpeg, signature, username, blurry, artist name, too many ears, extra ear, poorly drawn face, deformed, disfigured, extra limb, ugly, horror, out of focus, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, poorly drawn hands, fused fingers, too many fingers,'
    # negative_prompt = None
    show_grid(prompts=prompts, pipe=pipe, negative_prompt=negative_prompt, save_path='./examples/lora_res_ym.jpg',
              hn=4, wn=5, num_inference_steps=30, width=512, height=512)





