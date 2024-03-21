from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import DonutProcessor
from inference import InferenceClass
from time import time
from PIL import Image
import numpy as np
import torch
import os

model_path = f"HamAndCheese82/math_ocr_donut_onnx_v1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time()
model = ORTModelForVision2Seq.from_pretrained(
    model_path,
    use_cache=True,
    use_io_binding=True,
)
processor = DonutProcessor.from_pretrained(model_path)
print(f'loading duration: {time()-start_time}')  # 8.925255298614502 seconds

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

print(f"first inference duration: {results[0]['total_dur']}")  # 0.3307507038116455 seconds
print(f"mean inference duration: {np.mean([result['total_dur'] for result in results])}")  # 0.1715317964553833 seconds
print(f"mean inference duration without first: {np.mean([result['total_dur'] for result in results][1:])}")  # 0.15705735033208673 seconds


"""
2024-03-21 14:40:28.979640272 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] 
    Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. 
    e.g. ORT explicitly assigns shape related ops to CPU to improve perf.


2024-03-21 14:40:28.979674602 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] 
    Rerunning with verbose output on a non-minimal build will show node assignments.
"""
