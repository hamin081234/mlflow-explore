from ast import Tuple
import mlflow
from tempfile import TemporaryDirectory
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import DonutProcessor, PretrainedConfig
import os
import onnx
from onnxruntime import InferenceSession
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _add_code_from_conf_to_system_path, _get_flavor_configuration


def save_transformers_donut(
    # self,
    experiment_name,
    run_name,
    params,
    metrics,
    model,
    processor,
    model_name,
    task="image-to-text",
    tags=None,
):
    """
    HuggingFace Transformers 모델 중 image-to-text task를 위한 모델을
    mlflow에 학습 기록과 모델을 저장하기 위한 함수
    Args:
        experiment_name (str): mlflow experiment 프로젝트 이름 지정
        run_name (str): mlflow run 이름 지정
        params (dict): 모델의 최적 파라미터
        metrics (dict): 결과 점수(auc, acc 등)
        model (VisualEncoderDecoderModel): 학습된 모델,
        processor : 모델의 전처리 또는 후처리를 맡는 클래스
        model_name (str): 저장할 모델 이름
        task (str): huggingface에서 지정한 task 파이프라인 이름
        tags (dict): mlflow run의 태그값
    """
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # signature = infer_signature(signature_data, model.predict(signature_data))
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": processor.tokenizer,
                "image_processor": processor.image_processor,
            },
            processor=processor,
            artifact_path=run_name,
            task=task,
            registered_model_name=model_name,
        )


def load_transformers_donut(model_name, alias=None, model_version=None):
    """
    Model registry에 저장된 모델 불러오기
    Args:
        model_name (str): 모델 이름
        alias (str): 모델에 태깅된 @alias
        model_version (str): 모델 버전
    Returns:
        mlflow에 저장된 모델
    """
    if alias is not None:
        model_uri = f"models:/{model_name}@{alias}"
    elif model_version is not None:
        model_uri = f"models:/{model_name}/{model_version}"
    else:
        raise ValueError("Specify either model_version or alias")

    model = mlflow.transformers.load_model(
        model_uri=model_uri,
        return_type = 'components'
    )
    processor = DonutProcessor(image_processor=model['image_processor'], tokenizer=model['tokenizer'])
    temp_dir = TemporaryDirectory()
    model['model'].save_pretrained(temp_dir.name)

    model = ORTModelForVision2Seq.from_pretrained(temp_dir.name, from_transformers=True)
    temp_dir.cleanup()
    return model, processor


def save_onnx_donut(
    # self,
    experiment_name,
    run_name,
    params,
    metrics,
    model,
    processor,
    model_name,
    tags=None,
):
    """
    HuggingFace Transformers 모델을 onnx로 개별로 mlflow에 저장하는 함수
    Args:
        experiment_name (str): mlflow experiment 프로젝트 이름 지정
        run_name (str): mlflow run 이름 지정
        params (dict): 모델의 최적 파라미터
        metrics (dict): 결과 점수(auc, acc 등)
        model (ORTFor): 학습된 모델,
        processor : 모델의 전처리 또는 후처리를 맡는 클래스,
        model_name (str): 저장할 모델들의 앞에 붙을 이름
        tags (dict): mlflow run의 태그값
    """
    temp_dir = TemporaryDirectory()
    model.save_pretrained(temp_dir.name)
    processor.save_pretrained(temp_dir.name)

    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # signature = infer_signature(signature_data, model.predict(signature_data))
        for file in os.listdir(temp_dir.name):
            extension = file.split(".")[1]
            if extension == "onnx":
                model = onnx.load(temp_dir.name + "/" + file)
                onnx.checker.check_model(model)

                model_filename = file.split(".")[0]
                mlflow.onnx.log_model(
                    onnx_model=model,
                    artifact_path=run_name + "_" + model_name + "_" + model_filename,
                    registered_model_name=model_name + "_" + model_filename,
                )
            else:
                mlflow.log_artifact(
                    local_path=temp_dir.name + "/" + file, artifact_path="extra"
                )
    temp_dir.cleanup()


def load_model_path(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name='onnx')
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return onnx_model_artifacts_path


def load_onnx_donut(
    model_name, 
    run,
    alias=None, 
    model_version=None
):
    """
    Model registry에 저장된 모델 불러오기
    Args:
        model_name (str): 모델들 앞에 붙는 이름
        alias (str): 모델에 태깅된 @alias
        model_version (str): 모델 버전
    Returns:
        mlflow에 저장된 모델
    """
    model_filenames = ['decoder_model', 'encoder_model', 'decoder_with_past_model']

    model_uri_coll = {}
    if alias is not None:
        for filename in model_filenames:
            model_uri_coll[filename] = f"models:/{model_name}_{filename}@{alias}"
    elif model_version is not None:
        for filename in model_filenames:
            model_uri_coll[filename] = f"models:/{model_name}_{filename}/{model_version}"
    else:
        raise ValueError("Specify either model_version or alias")

    inf_sessions = {}
    for filename in model_uri_coll.keys():
        inf_sessions[filename] = InferenceSession(
            # decoder_model.SerializeToString(),
            # mlflow.artifacts.download_artifacts_from_uri(f"models:/decoder_model/{version}")+"/model.onnx",
            load_model_path(model_uri_coll[filename]),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    config = PretrainedConfig.from_pretrained(
        mlflow.artifacts.download_artifacts(
            f"runs:/{run}/etc/config.json"
        )
    )
    config.decoder = PretrainedConfig.from_dict(config.decoder)
    config.encoder = PretrainedConfig.from_dict(config.encoder)

    temp_model_save_dir = TemporaryDirectory()
    model = ORTModelForVision2Seq(
        encoder_session=inf_sessions['encoder_model'],
        decoder_session=inf_sessions['decoder_model'],
        decoder_with_past_session=inf_sessions['decoder_with_past_model'],
        onnx_paths=[temp_model_save_dir, temp_model_save_dir, temp_model_save_dir],
        config=config,
        model_save_dir=temp_model_save_dir
    )

    model_path = mlflow.artifacts.download_artifacts(f"runs:/{run}/extra/")
    processor = DonutProcessor.from_pretrained(model_path)
    
    return model, processor


def save_achitecture_donut(
    # self,
    experiment_name,
    run_name,
    params,
    metrics,
    model,
    processor,
    task="image-to-text",
    tags=None,
):
    """
    MLflow에 학습 기록과 모델 저장
    모델 종류에 따라 달라질 수 있기 때문에 수정 필요 TODO 수정 필요
    Args:
        experiment_name (str): mlflow experiment 프로젝트 이름 지정
        run_name (str): mlflow run 이름 지정
        params (dict): 모델의 최적 파라미터
        metrics (dict): 결과 점수(auc, acc 등)
        model (ORTFor): 학습된 모델,
        processor : 모델의 전처리 또는 후처리를 맡는 클래스
        task (str): huggingface에서 지정한 task 파이프라인 이름
        tags (dict): mlflow run의 태그값
    """
    temp_dir = TemporaryDirectory()
    model.save_pretrained(temp_dir.name)
    processor.save_pretrained(temp_dir.name)

    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # signature = infer_signature(signature_data, model.predict(signature_data))
        mlflow.log_artifacts(temp_dir.name, artifact_path="model")

    temp_dir.cleanup()


def load_architecture_donut(run):
    model_path = mlflow.artifacts.download_artifacts(f"runs:/{run}/model/")
    processor = DonutProcessor.from_pretrained(model_path)
    model = ORTModelForVision2Seq.from_pretrained(
        model_path, use_cache=True, use_io_binding=True
    )
    return model, processor
