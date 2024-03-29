import re
import time
from PIL import Image

from utils.metrics import character_error_rate, split_decoded_output, latex_word_error_rate

def load_img(path):
    """
    이미지가 저장되어 있는 경로를 받고, 그 이미지를 PIL.Image 객체로 변환해서 반환하는 함수

    :param path (str): 이미지가 저장된 경로

    :return img (PIL.Image): 변환된 PIL.Image 객체
    :return dur (float): 이미지를 불러오기까지 걸린 시간
    """
    dur = time.time()
    img = Image.open(path).convert("RGB")
    dur = time.time() - dur

    return img, dur


class InferenceClass:
    """
    모델이 추론하는 과정에 필요한 함수들을 지니고 있는 클래스
    :param model (VisualEncoderDecoderModel): 추론을 해주는 Donut 모델
    :param processor (DonutProcessor): 전처리와 후처리를 해주는 DonutProcessor 객체
    :param device (str): cuda 또는 cpu
    """
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

        self.model.to(self.device)

    def preprocess_img(self, images):
        """
        이미지를 전처리 해주는 함수

        :param images (List[PIL.Image]): 이미지들을 담고 있는 리스트

        :return pixel_values (torch.Tensor): 모델의 input size에 맞게 변환된 이미지
        :return dur (float): 변환시키는 동안 걸린 시간
        """
        images = images.convert("RGB")
        dur = time.time()
        pixel_values = self.processor(images, return_tensors='pt').pixel_values
        dur = time.time() - dur

        return pixel_values, dur

    def decode_tokens(self, sequences):
        """
        모델이 출력한 token_id 리스트를 decode해서 글 종류 여부와 사용 할 수 있는 latex 코드로 반환

        :param sequences (List[str]): 모델이 출력한 token_id 리스트

        :return pred_class_list (List[str]): 모델이 예측한 글 종류 (HW: 손글씨, P: 인쇄체)
        :return latex_list (List[str]): 모델의 출력값을 latex 코드로 변환
        :return dur (float): 디코딩까지 걸린 시간
        """
        dur = time.time()
        decoded_sequences = self.processor.batch_decode(sequences, skip_special_tokens=True)
        dur = time.time() - dur

        pred_class_list = []
        latex_list = []
        for output in decoded_sequences:
            pred_class = 'HW' if re.findall(r'<.*?>', output)[1] == '<s_HW>' else 'P'
            pred_class_list.append(pred_class)

            special_tokens = self.processor.tokenizer.added_tokens_decoder
            spc_rm = '|'.join([str(special_tokens[key]) for key in special_tokens.keys()])
            latex = re.sub(r"{}".format(spc_rm), " ", output).strip()
            latex_list.append(latex)

        return pred_class_list, latex_list, dur

    def generate_model_output(self, pixel_values, decoder_input_ids):
        """
        이미지와 task 종류를 입력 받고 모델의 추론 결과를 반환하는 함수

        :param pixel_values (torch.Tensor): DonutProcessor으로 전처리 된 이후의 이미지 값
        :param decoder_input_ids (torch.Tensor): 디코더에게 입력해 줄 task 시작 토큰

        :return outputs (dict): 모델이 추론한 결과를 담고 있는 dictionary
        :return dur (float): 모델이 추론하는 동안 걸린 시간
        """

        dur = time.time()
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
        dur = time.time() - dur

        return outputs, dur

    def inference(self, images, task_prompt="<MathOCR>"):
        """
        모델이 추론한 입력한 이미지에 대한 결과를 후처리한 이후의 결과와 그 과정 중에 걸린 시간을 기록해서 반환하는 함수
        
        :param images (PIL.Image): 추론 할 이미지
        :param task_prompt (str): 모델이 하게 될 task를 알려주는 토큰 (수식 OCR에 사용하기 때문에 default로 <MathOCR>)
        
        :return output (dict):
            img_process_dur (float): 이미지 전처리 소요시간,
            inference_dur (float): 추론 소요시간,
            decode_dur (float): decoding 소요시간,
            total_dur (float): 총 소요시간,
            pred_class (str): 글 종류,
            pred_latex (str): 이미지 속 수식들에 대한 latex 코드
        """
        # images = images.convert("RGB")

        pixel_values, img_process_dur = self.preprocess_img(images)

        # 추가 수정이 필요 (batch size > 1 일때도 대응 할 수 있도록)
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]

        outputs, inference_dur = self.generate_model_output(pixel_values.to(self.device), decoder_input_ids.to(self.device))
        pred_class_list, latex_list, decode_dur = self.decode_tokens(outputs.sequences)

        output = {
            'img_process_dur': img_process_dur,
            'inference_dur': inference_dur,
            'decode_dur': decode_dur,
            'total_dur': img_process_dur + inference_dur + decode_dur,
            'pred_class': pred_class_list,
            'pred_latex': latex_list
        }

        return output

    def evaluate(self, img, gt_latex_list):
        """
        입력 받은 이미지에 대해 추론을 잘 하는지 스코어들을 기록해주는 함수
        
        :param img (PIL.Image): 추론 할 이미지
        :param gt_latex_list (List[str]): 이미지 속에 존재하는 ground truth latex 코드
        
        :return output (dict):
            inference 함수에서 출력된 값 +
            gt_latex (str): ground truth의 latex 코드
            gt_list (List(str)): ground truth를 단어 단위로 나눈 리스트
            pred_list (List(str)): 예측한 latex 코드를 단어 단위로 나눈 리스트
            char_edit_distance (int): gt_latex와 pred_latex를 글자 단위로 보았을 때의 차이
            cer (float): gt_latex와 pred_latex의 character error rate
            word_edit_distance (int): gt_latex와 pred_list를 단어 단위로 보았을 때의 차이
            gt_word_len (int): grount truth의 latex 코드의 단어 개수
            wer (float): gt_list와 pred_list의 word error rate
        """
        output = self.inference(img)

        scores = []
        for gt_latex, pred_latex in zip(gt_latex_list, output['pred_latex']):
            char_distance, cer = character_error_rate(gt_latex, pred_latex)
            gt_list = split_decoded_output(gt_latex)
            pred_list = split_decoded_output(pred_latex)
            word_distance, wer = latex_word_error_rate(gt_list, pred_list)
            score = {
                'char_distance': char_distance,
                'cer': cer,
                'gt_list': gt_list,
                'gt_word_len': len(gt_list),
                'pred_list': pred_list,
                'word_distance': word_distance,
                'wer': wer,
            }
            scores.append(score)

        output['gt_latex'] = gt_latex_list
        output['gt_list'] = [score['gt_list'] for score in scores]
        output['pred_list'] = [score['pred_list'] for score in scores]
        output['char_edit_distance'] = [score['char_distance'] for score in scores]
        output['cer'] = [score['cer'] for score in scores]
        output['word_edit_distance'] = [score['word_distance'] for score in scores]
        output['gt_word_len'] = [score['gt_word_len'] for score in scores]
        output['wer'] = [score['wer'] for score in scores]

        return output