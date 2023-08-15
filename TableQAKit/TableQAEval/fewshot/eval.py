import json
import argparse
import re
import string

import jieba
from fuzzywuzzy import fuzz
import difflib
import numpy as np

from typing import List
from collections import Counter
# from rouge import Rouge

def is_numeric(string):
    try:
        float(string)
        return True
    except ValueError:
        return False




def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    if is_numeric(prediction) and is_numeric(ground_truth):
        if float(prediction) == float(ground_truth):
            return 1
        else:
            return 0
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


parser = argparse.ArgumentParser()


parser.add_argument('--pred_file', type=str, default='../results/fewshot/chatglm2_results.json')
# parser.add_argument('--gold_file', type=str, default='../data/sql_fourshot.json')
args = parser.parse_args()


# with open(args.gold_file, 'r') as file:
    # gold_data = json.load(file)  
    
with open(args.pred_file, 'r') as file:
    pred_data = json.load(file)  
    
scores = []
for item in pred_data:
    score = qa_f1_score(item['pred'], item['gold'])
    scores.append(score)

print(sum(scores) / len(scores))

