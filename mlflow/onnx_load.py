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

import torch

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _add_code_from_conf_to_system_path, _get_flavor_configuration

import os

def load_model_path(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name='onnx')
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return onnx_model_artifacts_path


version = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
run = '3ea7bb45abd34d099c8b0ca0f4efce64'
experiment_name = f"donut_decoder_onnx_trial"
experiment_id = mlflow.set_experiment(experiment_name).experiment_id

inf_sessions = {}
for model_name in ['decoder_model', 'encoder_model', 'decoder_with_past_model']:
    start_time = time()
    inf_sessions[model_name] = InferenceSession(
        # decoder_model.SerializeToString(),
        # mlflow.artifacts.download_artifacts_from_uri(f"models:/decoder_model/{version}")+"/model.onnx",
        load_model_path(f"models:/{model_name}/{version}"),
        providers=["CUDAExecutionProvider"],
    )
    print(f'{model_name} donwload_artifacts: {time()-start_time}')

config = PretrainedConfig.from_pretrained(
    mlflow.artifacts.download_artifacts(
        f"runs:/{run}/etc/config.json"
    )
)
config.decoder = PretrainedConfig.from_dict(config.decoder)
config.encoder = PretrainedConfig.from_dict(config.encoder)

temp_model_save_dir = TemporaryDirectory()
ort_model = ORTModelForVision2Seq(
    encoder_session=inf_sessions['encoder_model'],
    decoder_session=inf_sessions['decoder_model'],
    decoder_with_past_session=inf_sessions['decoder_with_past_model'],
    onnx_paths=[temp_model_save_dir, temp_model_save_dir, temp_model_save_dir],
    config=config,
    model_save_dir=temp_model_save_dir
)
model_path = mlflow.artifacts.download_artifacts(
    f"runs:/{run}/etc/"
)

processor = DonutProcessor.from_pretrained(model_path)
processor.image_processor.size = {'height': 1280, 'width': 960}
inf_cl = InferenceClass(model=ort_model, processor=processor, device=device)

for filename in os.listdir('images'):
    img = Image.open('images/'+filename).convert("RGB")
    print(inf_cl.inference(img))
