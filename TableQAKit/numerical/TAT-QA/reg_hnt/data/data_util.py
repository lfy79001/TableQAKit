import numpy as np
from tatqa_utils import to_number, is_number
import re
from enum import IntEnum

# "SPAN-TABLE-TEXT": 2 has no data
OPERATOR_CLASSES_ = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "CHANGE_RATIO": 3,
                    "AVERAGE": 4, "COUNT": 5, "SUM": 6, "DIFF": 7, "TIMES": 8, "DIVIDE": 9}


OPERATOR = ['+', '-', '*', '/']

SCALE = ["", "thousand", "million", "billion", "percent"]

def get_operators(derivation:str):
    res = []
    for c in derivation:
        if c in OPERATOR:
            res.append(c)
    return res

def get_operands(derivation):
    num_strs = re.split('\+|-|\*|/', derivatfion)
    result = []
    for it in num_strs:
        one = to_number(it)
        if one is not None:
            result.append(one)
    return result

def facts_to_nums(facts):
    return [to_number(f) for f in facts]

def _is_average(num_facts:list, answer):
    return round(np.average(num_facts), 2) == round(answer, 2)

def _is_change_ratio(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        ori_percent = round(100 * (num_facts[0] - num_facts[1]) / num_facts[1], 2)
        cands.append(ori_percent)
    if num_facts[0] != 0:
        ori_percent = round(100 * (num_facts[1] - num_facts[0]) / num_facts[0], 2)
        cands.append(ori_percent)
    return round(answer, 2) in cands 

def _is_division(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        cands.append(round(num_facts[0]/num_facts[1], 2))
        cands.append(100 * round(num_facts[0]/num_facts[1], 2))
    if num_facts[0] != 0:
        cands.append(round(num_facts[1]/num_facts[0], 2))
        cands.append(100 * round(num_facts[1]/num_facts[0], 2))
    return round(answer, 2) in cands

def _is_diff(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    ans_1 = round(num_facts[0] - num_facts[1], 2)
    ans_2 = round(num_facts[1] - num_facts[0], 2)
    return round(answer, 2) in (ans_1, ans_2)

def _is_sum(num_facts:list, answer):
    return round(np.sum(num_facts), 2) == round(answer, 2)

def _is_times(num_facts:list, answer):
    return round(np.prod(num_facts), 2) == round(answer, 2)

def get_operator_class(derivation:str, answer_type:str, facts:list, answer, mapping:dict, scale, OPERATOR_CLASSES):
    operator_class = None
    if answer_type == "span":
        if "table" in mapping:
            operator_class = OPERATOR_CLASSES["SPAN-TABLE"]
        else:
            operator_class = OPERATOR_CLASSES["SPAN-TEXT"]
    elif answer_type == "multi-span":
        operator_class = OPERATOR_CLASSES["MULTI_SPAN"]
    elif answer_type == "count":
        operator_class = OPERATOR_CLASSES["COUNT"]
    elif answer_type == "arithmetic":
        num_facts = facts_to_nums(facts)
        if not is_number(str(answer)):
            return None  # not support date
        if _is_change_ratio(num_facts, answer):
            operator_class = OPERATOR_CLASSES["CHANGE_RATIO"]
        elif _is_average(num_facts, answer):
            operator_class = OPERATOR_CLASSES["AVERAGE"]
        elif _is_sum(num_facts, answer):
            operator_class = OPERATOR_CLASSES["SUM"]
        elif _is_times(num_facts, answer):
            operator_class = OPERATOR_CLASSES["TIMES"]
        elif _is_diff(num_facts, answer):
            operator_class = OPERATOR_CLASSES["DIFF"]
        elif _is_division(num_facts, answer):
            operator_class = OPERATOR_CLASSES["DIVIDE"]

    return operator_class