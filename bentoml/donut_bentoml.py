import bentoml
from mlflow_manager import load_transformers_donut
# from transformers import VisionEncoderDecoderModel, DonutProcessor

# processor = DonutProcessor.from_pretrained("HamAndCheese82/math_ocr_donut_v1")
# model = VisionEncoderDecoderModel.from_pretrained("HamAndCheese82/math_ocr_donut_v1")

model, processor = load_transformers_donut("donut_transformers", model_version=1)

bentoml.transformers.save_model("donut_processor", processor)
bentoml.transformers.save_model("donut_model", model)
