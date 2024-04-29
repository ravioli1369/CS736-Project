import os
from glob import glob

models = glob("checkpoint/*0.pth")
for i, model in enumerate(models):
    print(model.split("/")[-1], f"{i}/{len(models)}")
    num_iters = int(model.split("_")[-1].split(".")[0])
    # if not (num_iters % 10000 == 0) | ((num_iters % 1000 == 0) & (num_iters <= 10000)):
    #    os.system(f"rm {model}")
    n = 30
    path = f"checkpoint/finetune/{num_iters}+{num_iters//n}/"
    if not os.path.exists(path + "g_final.pth"):
        os.makedirs(path, exist_ok=True)
        cmd = f"python run.py --data_root data/covid-chestxray-dataset/resized/ --mask_root data/covid-chestxray-dataset/resized/masks/1_hole/ --finetune --model_path {model} --num_iters {int(num_iters + num_iters//n)} --model_save_path {path}"
        print(cmd)
        os.system(cmd)
