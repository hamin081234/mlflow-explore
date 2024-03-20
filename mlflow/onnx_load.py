from PIL import Image
from inference import InferenceClass
import torch
import os
from mlflow_manager import load_onnx_donut

model_version = 3
run = '3ea7bb45abd34d099c8b0ca0f4efce64'
model_name = 'donut'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, processor = load_onnx_donut(
    model_name=model_name,
    model_version=model_version,
    run=run
)

processor.image_processor.size = {'height': 1280, 'width': 960}
inf_cl = InferenceClass(model=model, processor=processor, device=device)

for filename in os.listdir('images'):
    img = Image.open('images/'+filename).convert("RGB")
    print(inf_cl.inference(img))
