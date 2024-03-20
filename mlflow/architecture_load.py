import mlflow
from responses import start
from transformers import DonutProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
from PIL import Image

import torch

from inference import InferenceClass
from time import time
import os

import onnxruntime
from mlflow_manager import load_architecture_donut

run = "322fd83bbd534976bb5b5ee1656cb985"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = load_architecture_donut(run)

processor.image_processor.size = {'height': 1280, 'width': 960}
inf_cl = InferenceClass(model=model, processor=processor, device="cuda")

for filename in os.listdir('images'):
    img = Image.open('images/'+filename).convert("RGB")
    print(inf_cl.inference(img))