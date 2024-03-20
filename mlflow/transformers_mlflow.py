import mlflow
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    T5ForConditionalGeneration,
)
from optimum.onnxruntime import ORTModelForVision2Seq

experiment_name = f"mlflow_transformers_trial"
tags = None
run_name = "mathocr_save"
params = None
metrics = None

model_path = f"HamAndCheese82/math_ocr_donut_v1"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
onnx = ""

# model_path = "HamAndCheese82/math_ocr_donut_v1"
task = "image-to-text"

processor = DonutProcessor.from_pretrained(model_path)

experiment_id = mlflow.set_experiment(experiment_name).experiment_id
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name=run_name,
    tags=tags,
) as run:
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": processor.tokenizer,
            "image_processor": processor.image_processor,
        },
        processor=processor,
        artifact_path=run_name,
        task=task,
    )
