from transformers import DonutProcessor, VisionEncoderDecoderModel
from mlflow_manager import save_transformers_donut

experiment_name = f"donut_mlflow_trial"
tags = {'test': 'yes'}
run_name = "transformers"
params = {'key': 'value'}
metrics = {'acc': 0.8}
task = "image-to-text"
model_name="donut_transformers"

model_path = f"HamAndCheese82/math_ocr_donut_v1"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
processor = DonutProcessor.from_pretrained(model_path)

save_transformers_donut(
    experiment_name=experiment_name,
    run_name=run_name,
    metrics=metrics,
    model=model,
    model_name=model_name,
    params=params,
    processor=processor,
    task=task,
    tags=tags
)