import mlflow
from transformers import DonutProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

from PIL import Image

from tempfile import TemporaryDirectory
from inference import InferenceClass

from mlflow_manager import load_transformers_donut

model_name = "donut_transformers"
version = 1

model, processor = load_transformers_donut(model_name=model_name, version=version)

inf_cl = InferenceClass(model=model, processor=processor, device='cpu')

img = Image.open('./bRXmz1.jpg').convert("RGB")

print(inf_cl.inference(img))

