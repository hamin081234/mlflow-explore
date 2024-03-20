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

run = "322fd83bbd534976bb5b5ee1656cb985"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = mlflow.artifacts.download_artifacts(f"runs:/{run}/model/")

start_time = time()
processor = DonutProcessor.from_pretrained(model_path)
processor.image_processor.size = {"height": 1280, "width": 960}
ort_model = ORTModelForVision2Seq.from_pretrained(
    model_path, use_cache=True, use_io_binding=True
)
print(f"model loading time: {time()-start_time}")

inf_cl = InferenceClass(model=ort_model, processor=processor, device="cuda")

for filename in os.listdir('images'):
    img = Image.open('images/'+filename).convert("RGB")
    print(inf_cl.inference(img))