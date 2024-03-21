from transformers import DonutProcessor, VisionEncoderDecoderModel
from inference import InferenceClass
from time import time
from PIL import Image
import numpy as np
import torch
import os

model_path = f"HamAndCheese82/math_ocr_donut_v1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time()
model = VisionEncoderDecoderModel.from_pretrained(model_path)
processor = DonutProcessor.from_pretrained(model_path)
print(f'loading duration: {time()-start_time}')  # 2.8561253547668457 seconds

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

print(f"first inference duration: {results[0]['total_dur']}")  # 0.505608081817627 seconds
print(f"mean inference duration: {np.mean([result['total_dur'] for result in results])}")  # 0.17864364385604858 seconds
print(f"mean inference duration without first: {np.mean([result['total_dur'] for result in results][1:])}")  # 0.14891960404135965 seconds
