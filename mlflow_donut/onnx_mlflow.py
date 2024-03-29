from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import DonutProcessor

from mlflow_manager import save_onnx_donut

experiment_name = f"donut_mlflow_trial"
tags = None
run_name = "onnx"
params = {'key': 'value'}
metrics = {'acc': 0.8}
model_name = "donut"


model_path = f"HamAndCheese82/math_ocr_donut_onnx_v1"
model = ORTModelForVision2Seq.from_pretrained(
    model_path,
    use_cache=True,
    use_io_binding=True,
    provider="CPUExecutionProvider",
    # export=True
)
processor = DonutProcessor.from_pretrained(model_path)

save_onnx_donut(
    experiment_name=experiment_name,
    run_name=run_name,
    params=params,
    metrics=metrics,
    model=model,
    processor=processor,
    model_name=model_name
)
