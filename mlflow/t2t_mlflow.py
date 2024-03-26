import mlflow
from transformers import pipeline
from transformers import AutoProcessor
from transformers import AutoModelForCTC, TrainingArguments, Trainer

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

print(processor)

experiment_name = f"pipeline_trial"
tags = None
run_name = "xgb"
params = {'key': 'value'}
metrics = {'acc': 0.8}
model_name="wav2vec_model"


transcriber = pipeline(task="automatic-speech-recognition")

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
        transformers_model=transcriber,
        # transformers_model={
        #     "model": model,
        #     "feature_extractor": processor.feature_extractor,
        #     "tokenizer": processor.tokenizer
        #     # "processor": processor
        # },
        # processor=processor,
        artifact_path=run_name,
        registered_model_name=model_name,
        task="automatic-speech-recognition"
    )
