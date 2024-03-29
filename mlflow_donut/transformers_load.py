from PIL import Image
from inference import InferenceClass

import os
import torch
from time import time
from mlflow_manager import load_transformers_donut
import numpy as np

model_name = "donut_transformers"
model_version = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time()
model, processor = load_transformers_donut(model_name=model_name, model_version=model_version)
print(f"loading_time: {time()-start_time}")  # 5.641238689422607 seconds 6.169370412826538

processor.image_processor.size = {"height": 1280, "width": 960}
inf_cl = InferenceClass(model=model, processor=processor, device=device)

results = []
file_list = os.listdir('images')
file_list.append(file_list[0])
for filename in file_list:
    img = Image.open('images/'+filename).convert("RGB")
    out = inf_cl.inference(img)
    print(out)
    results.append(out)

print(f"first inference duration: {results[0]['total_dur']}")  # 0.4304921627044678 seconds 0.4823119640350342
print(f"mean inference duration: {np.mean([result['total_dur'] for result in results])}")  # 0.1822368303934733 seconds 0.17295968532562256
print(f"mean inference duration without first: {np.mean([result['total_dur'] for result in results][1:])}")  # 0.13258576393127441 seconds 0.14483675089749423
