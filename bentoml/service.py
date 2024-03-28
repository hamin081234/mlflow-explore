import bentoml
from bentoml.io import Image, JSON, File
from PIL import Image as PILImage
import re
from time import time
import io
import torch
from inference import InferenceClass

processor_ref = bentoml.transformers.get("donut_processor:latest")
model_ref = bentoml.transformers.get("donut_model:latest")

class DonutRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'cpu')
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.processor = bentoml.transformers.load_model(processor_ref)
        self.model = bentoml.transformers.load_model(model_ref)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

        self.inferece_class = InferenceClass(self.model, self.processor, self.device)

    @bentoml.Runnable.method(batchable=False)
    def generate_latex(self, inp: Image):
        print(self.device)

        inp = inp.convert("RGB")
        result = self.inferece_class.inference(inp)

        return result

donut_runner = bentoml.Runner(DonutRunnable, name="donut_runner", models=[processor_ref, model_ref])

svc = bentoml.Service("image2latex", runners=[donut_runner])

@svc.api(input=File(), output=JSON())
async def generate_latex(f):
    input_bytes = io.BytesIO(io.BytesIO(f.read()).read())
    img = PILImage.open(input_bytes)

    return await donut_runner.generate_latex.async_run(img)