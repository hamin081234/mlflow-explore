from json import encoder
import mlflow
import onnx, onnxruntime
from onnxruntime import InferenceSession
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    DonutImageProcessor,
    XLMRobertaTokenizerFast,
    PretrainedConfig,
)
from optimum.onnxruntime import ORTModelForVision2Seq
from PIL import Image

from inference import InferenceClass
from tempfile import TemporaryDirectory
from time import time

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _add_code_from_conf_to_system_path, _get_flavor_configuration

import os

def load_model_path(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name='onnx')
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return onnx_model_artifacts_path


version=2
experiment_name = f"donut_decoder_onnx_trial"
experiment_id = mlflow.set_experiment(experiment_name).experiment_id

# start_time = time()
# decoder_model = mlflow.onnx.load_model(f"models:/decoder_model/{version}")
# print(f'decode_model onnx.load_model: {time()-start_time}')

# start_time = time()
# encoder_model = mlflow.onnx.load_model(f"models:/encoder_model/{version}")
# print(f'encode_model onnx.load_model: {time()-start_time}')

# start_time = time()
# decoder_with_past_model = mlflow.onnx.load_model(f"models:/decoder_with_past_model/{version}")
# print(f'decode_past_model onnx.load_model: {time()-start_time}')

start_time = time()
decoder_is = InferenceSession(
    # decoder_model.SerializeToString(),
    # mlflow.artifacts.download_artifacts_from_uri(f"models:/decoder_model/{version}")+"/model.onnx",
    load_model_path(f"models:/decoder_model/{version}"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f'decode_model donwload_artifacts: {time()-start_time}')

start_time = time()
encoder_is = InferenceSession(
    # encoder_model.SerializeToString(),
    # mlflow.artifacts.download_artifacts_from_uri(f"models:/encoder_model/{version}")+"/model.onnx",
    load_model_path(f"models:/encoder_model/{version}"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f'encode_model donwload_artifacts: {time()-start_time}')

start_time = time()
decoder_with_past_is = InferenceSession(
    # decoder_with_past_model.SerializeToString(),
    # mlflow.artifacts.download_artifacts(f"models:/decoder_with_past_model/{version}"+"/model.onnx"),
    load_model_path(f"models:/decoder_with_past_model/{version}"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f'decode_past_model donwload_artifacts: {time()-start_time}')

config = PretrainedConfig.from_pretrained(
    mlflow.artifacts.download_artifacts(
        "runs:/de4c64f9d4a947dfaab453426101622b/etc/config.json"
    )
)
config.decoder = PretrainedConfig.from_dict(config.decoder)
config.encoder = PretrainedConfig.from_dict(config.encoder)
print(config.decoder.model_type)

print(mlflow.artifacts.download_artifacts(f"models:/decoder_model/{version}"))
print(mlflow.artifacts.download_artifacts(f"models:/encoder_model/{version}"))
print(mlflow.artifacts.download_artifacts(f"models:/decoder_with_past_model/{version}"))

temp_model_save_dir = TemporaryDirectory()
ort_model = ORTModelForVision2Seq(
    encoder_session=encoder_is,
    decoder_session=decoder_is,
    decoder_with_past_session=decoder_with_past_is,
    onnx_paths=[temp_model_save_dir, temp_model_save_dir, temp_model_save_dir],
    config=config,
    model_save_dir=temp_model_save_dir
)

# print(ort_model)

print(
    mlflow.artifacts.download_artifacts("runs:/de4c64f9d4a947dfaab453426101622b/etc/")
)

model_path = mlflow.artifacts.download_artifacts(
    "runs:/de4c64f9d4a947dfaab453426101622b/etc/"
)

processor = DonutProcessor.from_pretrained(model_path)
# processor.image_processor.size={'height':1280, 'width':920}

inf_cl = InferenceClass(model=ort_model, processor=processor, device="cpu")

img = Image.open("./bRXmz1.jpg").convert("RGB")

print(inf_cl.inference(img))
