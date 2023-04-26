# coding:utf-8
# @Email: wangguisen@donews.com
# @Time: 2023/4/18 15:48
# @File: down_demo.py

import sys
import os
import json

if __name__ == "__main__":
    jsonfile = "../data/pokemon/pokeman.json"  # sys.argv[1]
    savedir = "../data/pokemon/pokemon-blip-captions"  # sys.argv[2]
    os.system("curl -X GET \"https://datasets-server.huggingface.co/first-rows?dataset=lambdalabs%2Fpokemon-blip-captions&config=lambdalabs--pokemon-blip-captions&split=train\" > ../data/pokemon/pokeman.json")
    txtfile = savedir.rstrip("/") + "_text.txt"
    with open(jsonfile) as fb:
        lines = fb.readlines()
        line = lines[0].rstrip()
        data_dict = eval(line)
        # data_dict = json.load(fb)
    print(data_dict.keys())

    txt_list = []
    for ii in data_dict["rows"]:
        idx = ii["row_idx"]
        url = ii["row"]["image"]["src"]
        os.system(f"wget \"{url}\" -O {savedir}/{idx}.jpg")
        txt = ii["row"]["text"]
        txt_list.append(txt)
        with open(f"{savedir}/{idx}.txt", 'w') as fb:
            fb.writelines(txt)
    with open(txtfile, 'w') as fb:
        fb.writelines("\n".join(txt_list))






