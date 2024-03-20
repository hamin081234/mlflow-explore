import bentoml
import transformers
from transformers import DonutProcessor, VisionEncoderDecoderModel, pipeline

model_path = "HamAndCheese82/math_ocr_donut_v1"

processor = DonutProcessor.from_pretrained(model_path)
processor.image_processor.size = {'height': 1280, 'width': 960}

donut_model = VisionEncoderDecoderModel.from_pretrained(model_path)

task = "image-to-text"

mm_pipeline = pipeline(task, model=model_path)

bentoml.transformers.save_model(
    task,
    transformers.pipeline(task, model=model_path),
    metadata=dict(model_name=model_path),
)
