import bentoml
from utils.mlflow_manager import load_transformers_donut

def save_mlflow_donut_to_bentoml(
    mlflow_name,
    model_bento_name,
    processor_bento_name,
    mlflow_alias=None,
    mlflow_version=None,
):
    if mlflow_alias:
        model, processor = load_transformers_donut(
            mlflow_name, model_alias=mlflow_alias
        )
    if mlflow_version:
        model, processor = load_transformers_donut(
            mlflow_name, model_version=mlflow_version
        )
    else:
        raise ValueError("Please enter either model's alias or version")

    bentoml.transformers.save_model(model_bento_name, model)
    bentoml.transformers.save_model(processor_bento_name, processor)


mlflow_name = "donut_transformers"
model_bento_name = "donut_model"
processor_bento_name = "donut_processor"
model_version = 1

save_mlflow_donut_to_bentoml(
    mlflow_name, model_bento_name, processor_bento_name, mlflow_version=model_version
)
