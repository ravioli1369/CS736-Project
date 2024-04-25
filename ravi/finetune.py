import os
from glob import glob

models = glob("checkpoint/model2/*.pth")
for (i, model) in enumerate(models):
    print(model.split("/")[-1], f"{i}/{len(models)}")
    num_iters = int(model.split("_")[-1].split(".")[0])
    #if not (num_iters % 10000 == 0) | ((num_iters % 1000 == 0) & (num_iters <= 10000)):
    #    os.system(f"rm {model}")
    path = f"checkpoint/model2/finetune/{num_iters}+{num_iters//10}/"
    os.makedirs(path, exist_ok=True)
    #print(f"python run.py --data_root data/covid-chestxray-dataset/resized/ --mask_root data/covid-chestxray-dataset/resized/masks/2_holes/ --n_threads 4 --finetune --model_path {model} --num_iters {num_iters//2} \n\n")
    os.system(f"python run.py --data_root data/covid-chestxray-dataset/resized/ --mask_root data/covid-chestxray-dataset/resized/masks/1_hole/ --finetune --model_path {model} --num_iters {int(num_iters + num_iters//10)} --model_save_path {path}")
