import mlflow
from xgboost import XGBClassifier

experiment_name = f"xgb_mlflow_trial"
tags = None
run_name = "xgb"
params = {'key': 'value'}
metrics = {'acc': 0.8}
model_name="xgb_model"

model = XGBClassifier()
xgb = model.fit([[0, 2]], [0])

experiment_id = mlflow.set_experiment(experiment_name).experiment_id
with mlflow.start_run(
    experiment_id=experiment_id,
    run_name=run_name,
    tags=tags,
) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.set_tag("run_id", run.info.run_id)
    # signature = infer_signature(signature_data, model.predict(signature_data))
    mlflow.xgboost.log_model(model, registered_model_name=model_name, artifact_path='xgb')

