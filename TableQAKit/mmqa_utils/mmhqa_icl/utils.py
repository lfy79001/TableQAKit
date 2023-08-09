"""
General utilities.
"""
import json
import os
from typing import List, Union, Dict
from functools import cmp_to_key
import math
from collections.abc import Iterable

from datasets import load_dataset

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def get_type(type_str):
    if type_str in ['ImageQ', 'ImageListQ']:
        return 'image'
    elif type_str in ['TableQ']:
        return 'table'
    elif type_str in ['TextQ']:
        return 'text'
    else:
        return 'compose'

def majority_vote(
        nsqls: List,
        pred_answer_list: List,
        allow_none_and_empty_answer: bool = False,
        allow_error_answer: bool = False,
        answer_placeholder: Union[str, int] = '<error|empty>',
        vote_method: str = 'prob',
        answer_biased: Union[str, int] = None,
        answer_biased_weight: float = None,
):
    """
    Determine the final nsql execution answer by majority vote.
    """

    def _compare_answer_vote_simple(a, b):
        """
        First compare occur times. If equal, then compare max nsql logprob.
        """
        if a[1]['count'] > b[1]['count']:
            return 1
        elif a[1]['count'] < b[1]['count']:
            return -1
        else:
            if a[1]['nsqls'][0][1] > b[1]['nsqls'][0][1]:
                return 1
            elif a[1]['nsqls'][0][1] == b[1]['nsqls'][0][1]:
                return 0
            else:
                return -1

    def _compare_answer_vote_with_prob(a, b):
        """
        Compare prob sum.
        """
        return 1 if sum([math.exp(nsql[1]) for nsql in a[1]['nsqls']]) > sum(
            [math.exp(nsql[1]) for nsql in b[1]['nsqls']]) else -1

    # Vote answers
    candi_answer_dict = dict()
    for (nsql, logprob), pred_answer in zip(nsqls, pred_answer_list):
        if allow_none_and_empty_answer:
            if pred_answer == [None] or pred_answer == []:
                pred_answer = [answer_placeholder]
        if allow_error_answer:
            if pred_answer == '<error>':
                pred_answer = [answer_placeholder]

        # Invalid execution results
        if pred_answer == '<error>' or pred_answer == [None] or pred_answer == []:
            continue
        if candi_answer_dict.get(tuple(pred_answer), None) is None:
            candi_answer_dict[tuple(pred_answer)] = {
                'count': 0,
                'nsqls': []
            }
        answer_info = candi_answer_dict.get(tuple(pred_answer), None)
        answer_info['count'] += 1
        answer_info['nsqls'].append([nsql, logprob])

    # All candidates execution errors
    if len(candi_answer_dict) == 0:
        return answer_placeholder, [(nsqls[0][0], nsqls[0][-1])]

    # Sort
    if vote_method == 'simple':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'prob':
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_with_prob), reverse=True)
    elif vote_method == 'answer_biased':
        # Specifically for Tabfact entailed answer, i.e., `1`.
        # If there exists nsql that produces `1`, we consider it more significant because `0` is very common.
        assert answer_biased_weight is not None and answer_biased_weight > 0
        for answer, answer_dict in candi_answer_dict.items():
            if answer == (answer_biased,):
                answer_dict['count'] *= answer_biased_weight
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    elif vote_method == 'lf_biased':
        # Assign weights to different types of logic forms (lf) to control interpretability and coverage
        for answer, answer_dict in candi_answer_dict.items():
            count = 0
            for nsql, _ in answer_dict['nsqls']:
                if 'map@' in nsql:
                    count += 10
                elif 'ans@' in nsql:
                    count += 10
                else:
                    count += 1
            answer_dict['count'] = count
        sorted_candi_answer_list = sorted(list(candi_answer_dict.items()),
                                          key=cmp_to_key(_compare_answer_vote_simple), reverse=True)
    else:
        raise ValueError(f"Vote method {vote_method} is not supported.")

    pred_answer_info = sorted_candi_answer_list[0]
    pred_answer, pred_answer_nsqls = list(pred_answer_info[0]), pred_answer_info[1]['nsqls']
    return pred_answer, pred_answer_nsqls


def load_data_split(dataset_to_load, split, data_dir=os.path.join(ROOT_DIR, 'datasets/')):
    if split == 'dev':
        split = 'validation'
    dataset_split_loaded = load_dataset(
        path=os.path.join(data_dir, "{}.py".format(dataset_to_load)), # specify a python builder to create a dataset
        streaming=True)[split]
    if dataset_to_load == 'mmqa': # DOING
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item['table']['page_title'] = data_item['table']['title']
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    else:
        raise ValueError(f'{dataset_to_load} dataset is not supported now.')
    return dataset_split_loaded


def pprint_dict(dic):
    print(json.dumps(dic, indent=2))


def flatten(nested_list):
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def parse_ans(ans_text, type, cot):
    # TODO: 增加对 python 的处理
    if not cot:
        return True, ans_text
    else:
        answer = ans_text.split("Answer: ")[1].strip()
        return True, answer