import mlflow
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    T5ForConditionalGeneration,
)
from optimum.onnxruntime import ORTModelForVision2Seq


is_onnx = True
use_cache = True

if is_onnx:
    # huggingface의 ORT 클래스는 mlflow에서 지원해주지 않음
    model_path = f"HamAndCheese82/math_ocr_donut_onnx_v1"
    model = ORTModelForVision2Seq.from_pretrained(
        model_path,
        use_cache=use_cache,
        use_io_binding=use_cache,
        provider="CPUExecutionProvider",
    )
    onnx = "onnx_"
else:
    model_path = f"HamAndCheese82/math_ocr_donut_v1"
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    onnx = ""

experiment_name = f"donut_{onnx}trial"
model_name = f"donut_{onnx}v1"
tags = None
run_name = "mathocr_v1"
params = None
metrics = None

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
