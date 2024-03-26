import bentoml

from bentoml.io import Text, Image
import re

processor_ref = bentoml.transformers.get("donut_processor:latest")
model_ref = bentoml.transformers.get("donut_model:latest")

class DonutRunnable(bentoml.Runnable):

    def __init__(self):
        self.processor = bentoml.transformers.load_model(processor_ref)
        self.model = bentoml.transformers.load_model(model_ref)
        # self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        # self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    @bentoml.Runnable.method(batchable=False)
    def generate_text(self, inp: Image):
        pixel_values = self.processor(inp, return_tensors='pt').pixel_values

        decoder_input_ids = self.processor.tokenizer(
            "<MathOCR>",
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.config.decoder.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )

        decoded_sequences = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)

        pred_class_list = []
        latex_list = []
        for output in decoded_sequences:
            pred_class = 'HW' if re.findall(r'<.*?>', output)[1] == '<s_HW>' else 'P'
            pred_class_list.append(pred_class)

            special_tokens = self.processor.tokenizer.added_tokens_decoder
            spc_rm = '|'.join([str(special_tokens[key]) for key in special_tokens.keys()])
            latex = re.sub(r"{}".format(spc_rm), " ", output).strip()
            latex_list.append(latex)

        return latex_list[0]

donut_runner = bentoml.Runner(DonutRunnable, name="donut_runner", models=[processor_ref, model_ref])
svc = bentoml.Service("image2latex", runners=[donut_runner])

@svc.api(input=Image(), output=Text())
async def generate_text(inp: Image):
    return await donut_runner.generate_speech.async_run(inp)
