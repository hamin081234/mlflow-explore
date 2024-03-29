import re
from typing import List
from nltk import edit_distance

def edit_distance_rate(gt, pred):
    """
    Donut Fine-tuning Tutorial에서 사용한 metric

    :param gt (str): ground truth 텍스트 문장
    :param pred (str): 예측한 텍스트 문장

    :return: 글자 단위로 계산한 두 문장의 차이의 비율
    """
    return edit_distance(pred, gt) / max(len(pred), len(gt))


def split_decoded_output(latex_str: str):
    """
    입력 받은 latex 코드를 지정한 단어 단위로 나눠주는 함수
    \a-zA-z*, $, {, }, [, ], (, ), _, ^, +, -, =, ,, :, ;, / 그리고 띄어쓰기를 기준으로 나뉨

    :param latex_str (str): latex 코드

    :return splitted (List[str]): 단어 단위로 나누어진 리스트
    """
    splitted = re.sub(r"\\displaystyle| ", "", latex_str).strip()
    deliminators = ['(\$)', '(\{)', '(\})', '(\[)', '(\])', '(\()', '(\))',
                    '(\_)', '(\^)', '(\+)', '(\-)', '(\=)', '(\,)', '(\:)', '(\;)', '(\/)', ' ']
    splitted = re.split(r"(\\[a-zA-Z]*)", splitted)
    temp = []
    for text in splitted:
        temp += re.split(r"{}".format('|'.join(deliminators)), text)
    splitted = [val for val in temp if val is not None and val != '']

    return splitted


def latex_word_error_rate(gt_list: List[str], pred_list: List[str]):
    """
    Latex 코드를 지정한 단어 단위로 나누어서 ground truth와 예측된 값의 word error rate을 구하는 함수

    :param gt_list (List(str)): gt_latex를 단어 단위로 나눈 리스트
    :param pred_list (List(str)): pred_latex를 단어 단위로 나눈 리스트

    :return distance (int): pred_list가 gt_list로 되기 위해 필요한 최소한의 편집 횟수
    :return wer (float): word error rate (distance를 ground truth의 단어 개수로 나눈 값)
    """

    distance = edit_distance(gt_list, pred_list)
    wer = distance / len(gt_list)

    return distance, wer


def character_error_rate(gt_latex: str, pred_latex: str):
    """
    ground truth latex 코드와 모델이 예측한 latex 코드의 character error rate을 구하는 함수
    :param gt_latex (str): ground truth latex 코드
    :param pred_latex (str): 모델이 추론한 latex 코드
    
    :return distance (int):
    :return cer (float):
    """
    distance = edit_distance(gt_latex, pred_latex)
    cer = distance / len(gt_latex)

    return distance, cer