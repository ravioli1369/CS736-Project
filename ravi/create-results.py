import os
from glob import glob

data_root = "data/covid-chestxray-dataset/resized/"
mask_root = data_root + "masks/4_holes/"
model_paths = glob("checkpoint/model4/finetune/**/g_final.pth", recursive=True)
for model_path in model_paths:
    test_cmd = f"python run.py --data_root {data_root} --mask_root {mask_root} --result_save_path {os.path.dirname(model_path)} --model_path {model_path} --test"
    print(test_cmd)
    os.system(test_cmd)
