from PIL import Image
from inference import InferenceClass

import os
import torch
from time import time
from mlflow_manager import load_transformers_onnx_donut
import numpy as np

model_name = "donut_transformers"
model_version = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time()
model, processor = load_transformers_onnx_donut(model_name=model_name, model_version=model_version)
print(f"loading_time: {time()-start_time}")  # 40.36700677871704 seconds 41.92785620689392

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

print(f"first inference duration: {results[0]['total_dur']}")  # 0.6825704574584961 seconds  0.6748528480529785
print(f"mean inference duration: {np.mean([result['total_dur'] for result in results])}")  # 0.34239665667215985 seconds 0.34968586762746173
print(f"mean inference duration without first: {np.mean([result['total_dur'] for result in results][1:])}")  # 0.27436189651489257 seconds 0.32012523304332385
