from PIL import Image

import torch

from inference import InferenceClass
from time import time
import os

from mlflow_manager import load_architecture_donut

import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-rn", "--run", dest="run_id", help="write run_id")
args = parser.parse_args()

run = args.run_id
device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time()
model, processor = load_architecture_donut(run)
print(f"loading_time: {time()-start_time}")  # 6.23365044593811 seconds 6.228323221206665

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

print(f"first inference duration: {results[0]['total_dur']}")  # 0.3457205295562744 seconds 0.3499622344970703
print(f"mean inference duration: {np.mean([result['total_dur'] for result in results])}")  # 0.17619220415751138 seconds 0.17982755104700723
print(f"mean inference duration without first: {np.mean([result['total_dur'] for result in results][1:])}")  # 0.1422865390777588 seconds 0.16436076164245605
