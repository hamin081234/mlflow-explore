import bentoml
from transformers import VisionEncoderDecoderModel, DonutProcessor

processor = DonutProcessor.from_pretrained("HamAndCheese82/math_ocr_donut_v1")
model = VisionEncoderDecoderModel.from_pretrained("HamAndCheese82/math_ocr_donut_v1")

bentoml.transformers.save_model("donut_processor", processor)
bentoml.transformers.save_model("donut_model", model)
