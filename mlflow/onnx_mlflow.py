import mlflow
import onnx, onnxruntime
import os
import json
from tempfile import TemporaryDirectory
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import DonutProcessor

temp_dir = TemporaryDirectory()
print(temp_dir.name)

model_path = f"HamAndCheese82/math_ocr_donut_onnx_v1"
model = ORTModelForVision2Seq.from_pretrained(
    model_path,
    use_cache=True,
    use_io_binding=True,
    provider="CPUExecutionProvider",
    # export=True
)
processor = DonutProcessor.from_pretrained(model_path)
model.save_pretrained(temp_dir.name)
processor.save_pretrained(temp_dir.name)

print(os.listdir(temp_dir.name))

experiment_name = f"donut_decoder_onnx_trial"
tags = None
run_name = "mathocr_v5"
params = None
metrics = None

experiment_id = mlflow.set_experiment(experiment_name).experiment_id
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name=run_name,
    tags=tags,
) as run:    
    for file in os.listdir(temp_dir.name):
        extension = file.split(".")[1]
        if extension == "onnx":
            model = onnx.load(temp_dir.name+'/'+file)
            model_name = file.split('.')[0]

            onnx.checker.check_model(model)
            mlflow.onnx.log_model(
                onnx_model=model["model"],
                artifact_path=run_name+'_'+model_name,
                registered_model_name=model_name,
            )
        else:
            mlflow.log_artifact(
                local_path=temp_dir.name+'/'+file, artifact_path="etc"
            )

temp_dir.cleanup()
