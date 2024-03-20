from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import DonutProcessor
from mlflow_manager import save_achitecture_donut

experiment_name = f"donut_mlflow_trial"
tags = None
run_name = "architecture"
params = {'key': 'value'}
metrics = {'acc': 0.8}

model_path = f"HamAndCheese82/math_ocr_donut_onnx_v1"
model = ORTModelForVision2Seq.from_pretrained(
    model_path,
    use_cache=True,
    use_io_binding=True,
    # export=True
)
processor = DonutProcessor.from_pretrained(model_path)
processor.image_processor.size = {"height": 1280, "width": 960}

save_achitecture_donut(
    experiment_name=experiment_name,
    run_name=run_name,
    params=params,
    metrics=metrics,
    model=model,
    processor=processor
)
