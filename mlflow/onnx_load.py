from PIL import Image
from inference import InferenceClass
import torch
import os
from mlflow_manager import load_onnx_donut

from time import time
import numpy as np
from argparse import ArgumentParser

model_version = 1
model_name = 'donut'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time()
model, processor = load_onnx_donut(
    model_name=model_name,
    model_version=model_version
)
print(f"loading_time: {time()-start_time}")  # 28.837988138198853 seconds 31.949495792388916

processor.image_processor.size = {'height': 1280, 'width': 960}
inf_cl = InferenceClass(model=model, processor=processor, device=device)

results = []
file_list = os.listdir('images')
file_list.append(file_list[0])
for filename in file_list:
    img = Image.open('images/'+filename).convert("RGB")
    out = inf_cl.inference(img)
    print(out)
    results.append(out)

print(f"first inference duration: {results[0]['total_dur']}")  # 0.36922740936279297 seconds 0.7241330146789551
print(f"mean inference duration: {np.mean([result['total_dur'] for result in results])}")  # 0.18675466378529867 seconds 0.2056848406791687
print(f"mean inference duration without first: {np.mean([result['total_dur'] for result in results][1:])}")  # 0.1502601146697998 seconds 0.15855318849736993
