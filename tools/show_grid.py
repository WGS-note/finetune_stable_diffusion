# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/3/27 15:45
# @File: show_grid.py
'''
show_grid
'''
import torch
from lora.lora_diffusion import image_grid
from lora.lora_diffusion import tune_lora_scale

def show_grid(prompts, pipe, negative_prompt=None, num_inference_steps=50, save_path='./examples/imgs.jpg', hn=2, wn=3, height=512, width=512):
    '''

    :param prompts:
    :param pipe:
    :param negative_prompt:
    :param num_inference_steps:
    :param save_path:
    :param hn:
    :param wn:
    :param height:
    :param width:
    :return:
    '''

    assert len(prompts) == hn * wn, "The number of images in the grid is the same as len(prompts)"

    outs = []
    for idx, prompt in enumerate(prompts):
        # torch.manual_seed(idx)
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5, height=height, width=width,).images[0]
        outs.append(image)

    imgs = image_grid(outs, hn, wn)
    imgs.save(save_path)

    return imgs

if __name__ == '__main__':

    prompts = ["best quality, masterpiece, 1girl, china dress, Beautiful face",
               "best quality, masterpiece, 1girl, china dress, Beautiful face",
               "1girl, kpop idol, yae miko, detached sleeves, bare shoulders, pink hair, long hair",
               "1girl, kpop idol, yae miko, detached sleeves, bare shoulders, pink hair, long hair",
               "A fairy flying in the sky, best quality, masterpiece, china dress, Beautiful face",
               "A fairy flying in the sky, best quality, masterpiece, china dress, Beautiful face", ]

    # show_grid(prompts=prompts, pipe=pipe, save_path='./examples/23.jpg', hn=2, wn=3)
