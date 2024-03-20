import mlflow
from transformers import DonutProcessor
from PIL import Image

from inference import InferenceClass

experiment_name = f"donut_decoder_onnx_trial"
experiment_id = mlflow.set_experiment(experiment_name).experiment_id

model = mlflow.transformers.load_model(
    f"runs:/c1f0ab4e1ad84821a9cb8bb28488de5f/mathocr_v5",
    return_type = 'components'
)

print('#'*100)
print(model['processor'])
print('#'*100)
print(model['image_processor'])
print('#'*100)
print(model['tokenizer'])

processor = DonutProcessor(image_processor=model['image_processor'], tokenizer=model['tokenizer'])

# inf_cl = InferenceClass(model=model['model'], processor=model['processor'], device='cpu')
inf_cl = InferenceClass(model=model['model'], processor=processor, device='cpu')

img = Image.open('./bRXmz1.jpg').convert("RGB")

print(inf_cl.inference(img))

