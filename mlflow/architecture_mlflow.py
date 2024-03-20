import mlflow
import onnx, onnxruntime
import os
import json
from tempfile import TemporaryDirectory
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import DonutProcessor

from inference import InferenceClass
from PIL import Image

experiment_name = f"mlflow_architecture_trial"
tags = None
run_name = "mathocr_save"
params = None
metrics = None


temp_dir = TemporaryDirectory()
print(temp_dir.name)

model_path = f"HamAndCheese82/math_ocr_donut_onnx_v1"
model = ORTModelForVision2Seq.from_pretrained(
    model_path,
    use_cache=True,
    use_io_binding=True,
    # export=True
)
print("#" * 100)
processor = DonutProcessor.from_pretrained(model_path)
processor.image_processor.size = {"height": 1280, "width": 960}
model.save_pretrained(temp_dir.name)
processor.save_pretrained(temp_dir.name)

experiment_id = mlflow.set_experiment(experiment_name).experiment_id
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name=run_name,
    tags=tags,
) as run:
    mlflow.log_artifacts(temp_dir.name, artifact_path="model")

temp_dir.cleanup()

inf_cl = InferenceClass(model=model, processor=processor, device="cuda")

img = Image.open("./bRXmz1.jpg").convert("RGB")

print(inf_cl.inference(img))
