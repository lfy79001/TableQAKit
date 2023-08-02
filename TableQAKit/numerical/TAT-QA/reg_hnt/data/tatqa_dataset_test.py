from collections import defaultdict
import re
import string
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import torch
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .file_utils import is_scatter_available
from tatqa_utils import  *
from .data_util import *
from .data_util import  _is_average, _is_change_ratio, _is_diff, _is_division, _is_sum, _is_times
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import dgl
from .pre_order import infix_to_prefix
from reg_hnt.reghnt.Vocabulary import *
from collections import defaultdict, OrderedDict
from itertools import groupby
from nltk.corpus import stopwords
stop_words = stopwords.words('english')




def get_order_by_tf_idf(question, paragraphs):
    sorted_order = []
    corpus = [question]
    for order, text in paragraphs.items():
        corpus.append(text)
        sorted_order.append(order)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x:x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]



def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def is_year(item):
    years = [str(i) for i in range(2000, 2051)]
    for year in years:
        if year in item:
            return True
    return False

def is_month(item):
    months = ['January','February','August','March',\
        'April','May','June','July','September',\
            'October','November','December']
    for month in months:
        if month in item:
            return True
    return False

def is_scale(item):
    item = sep_lower(item)
    scales = ['4rxr7y99', 'thousand', 'million', 'billion', 'percent', '%', '£m','USDm','EURm','€m','$m','$M','US$’000','$’000',"£’000",'£000','US$000','$000',"$'000"]
    for scale in scales:
        if scale in item:
            return True
    return False    
    
def is_None(item):
    if item == '' or item == 'N/A' or item == 'n/a' or item == '—' or item == '–' or item == '~'\
    or item == '$—' or item == '—%'or item =='$ —' or item == '— %' or item == "-" or item=='(%)'\
    or item == "$-"or item == "$ -" or item=="-%" or item=="- %"  or item == 'nm' or item=='nm ' or item=='n.m.'\
    or item=="*" or item=="$'000" or item=="$" or item=="%" or item == 'n/m' or item =='$’000'\
    or item=='Years' or item=='£m' or item=='USDm' or item=="EURm" or item == '€m' or item == '$m' or item=='$M'\
    or item=='US$’000' or item=="£’000" or item=='£000' or item=='$000' or item=='US$000' or item=='---':
        return True
    else:
        return False

# answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from}
def generate_class(answer_dict, mapping):
    operator_class = -1
    if answer_dict['answer_type'] == "span":
        if "table" in mapping:
            operator_class = OPERATOR_CLASSES["SPAN-TABLE"]
        else:
            operator_class = OPERATOR_CLASSES["SPAN-TEXT"]
    elif answer_dict['answer_type'] == "multi-span":
        operator_class = OPERATOR_CLASSES["MULTI_SPAN"]
    elif answer_dict['answer_type'] == "count":
        operator_class = OPERATOR_CLASSES["COUNT"]
    elif answer_dict['answer_type'] == "arithmetic":
        operator_class = OPERATOR_CLASSES["ARITHMETIC"]
    return operator_class

def string_tokenizer(string: str, tokenizer) -> List[int]:
    if not string:
        return []
    tokens = []
    string_subword_lens = []
    string_types = []
    prev_is_whitespace = True
    for i, c in enumerate(string):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)
    split_tokens = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer.tokenize(" " + token)
        else:
            sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
        string_subword_lens += [len(sub_tokens)]
    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return ids, string_subword_lens, tokens
'''
scale_type = ['', 'thousand', 'million', 'billion', 'percent'或'%']
'''

'''
table_cell_type_ids  0 none   1 date    2  numbers   3  row_head  4 column_head  5 danwei  -1不参与图网络的，参与训练
table_is_scale_type    0 none    1 scale
'''

def question_tokenize(question, tokenizer):
    if not question:
        return []
    tokens = []
    
    question_word_index = []
    question_subword_lens = []
    question_types = [0]
    question_word_types = []
    current_index = 1
    prev_is_whitespace = True

    question_number_value = []

    question = question.replace("?", '')
    for i, c in enumerate(question):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)
    split_tokens = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer.tokenize(" " + token)
        else:
            sub_tokens = tokenizer.tokenize(token)

        flag = 1
        subword_length = 0
        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            question_word_index.append(current_index)
            flag = 0
            subword_length += 1
            
        question_subword_lens += [subword_length]
        current_index += 1

        if is_number(token) and not is_year(token) and not is_month(tokens[i-1]):  
            question_number_value.append(float(to_number(token)))
            question_types += [2 for _ in range(len(sub_tokens))]
            question_word_types += [2]
        else:
            question_number_value.append(np.nan)
            question_types += [3 for _ in range(len(sub_tokens))]
            question_word_types += [3]

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    question_types += [0]    
    return ids, tokens, question_subword_lens, question_word_index, question_types, question_word_types,question_number_value


def clean_year(item):
    flag = 0
    years = [str(i) for i in range(2000, 2051)]
    num = 0
    this_year = ''
    for year in years:
        if year in item:
            num += 1
            this_year = year
    if num != 1 or item == '2015/16' or item == '2016/17' or item == '2017/18' or item == '2018/19' or item == '2019/20' or item =="2025-beyond":
        return item
    else:
        if is_month(item):
            return item
        else:
            if this_year != item:
                import pdb; pdb.set_trace()
            return this_year        

def scale_judge(item):
    item = sep_lower(item)
    scales = ['4rxr7y99', 'thousand', 'million', 'billion', 'percent', '%', '£m','USDm','EURm','€m','$m','$M','US$’000','$’000',"£’000",'£000','US$000','$000',"$'000"]
    for idx, scale in enumerate(scales):
        if scale in item:
            to_idx = 0
            if idx in [1, 12, 13, 14, 15, 16, 17, 18]:
                to_idx = 1
            elif idx in [2, 6, 7, 8, 9, 10, 11]:
                to_idx = 2
            elif idx in [3]:
                to_idx = 3
            elif idx in [4, 5]:
                to_idx = 4
            return idx, to_idx

def generate_number_scale(table, table_pd, table_cell_type_ids, table_is_scale_type, table_cell_number_value, table_cell_number_value_position, paragraphs):
    SCALE = ['', 'thousand', 'million', 'billion', 'percent']
    TMB_SCALE = ['thousand', 'million', 'billion']

    table_number_position = []
    for x, y in zip(table_cell_number_value_position, table_cell_number_value):
        if not np.isnan(y):
            table_number_position.append(x)
        else:
            table_number_position.append((-1, -1))
        
        
    table_cell_number_scale = list(map(lambda x: 0 if not np.isnan(x) else x, table_cell_number_value))
    global_scale = 0

    scale_position = np.where(table_is_scale_type > 0)
    for x, y in zip(*scale_position):
        
        this_scale, to_scale = scale_judge(table[x][y])
        if this_scale in [1, 2, 3, 6, 7]:
            global_scale = to_scale
            
            for idx in range(x, len(table)):
                for idy in range(len(table[idx])):
                    if (idx, idy) in table_cell_number_value_position:
                        table_cell_number_scale[table_cell_number_value_position.index((idx, idy))] = to_scale
        

    if global_scale == 0:
        for paragraph in paragraphs:
            paragraph_text = paragraph['text']
            if 'in thousand' in paragraph_text or 'In thousands' in paragraph_text:
                global_scale = 1
            if 'in million' in paragraph_text or 'In million' in paragraph_text:
                global_scale = 2
            if 'in billion' in paragraph_text or 'In billion' in paragraph_text:
                global_scale = 3
        if global_scale != 0:
            table_cell_number_scale = list(map(lambda x:global_scale if not np.isnan(x) else x, table_cell_number_scale))
            
    for x, y in zip(*scale_position):
        this_scale, to_scale = scale_judge(table[x][y])
        if this_scale in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            for item in table_number_position:
                if item[1] == y and item[0] > x:
                    table_cell_number_scale[table_cell_number_value_position.index(item)] = to_scale
                if item[0] == x and item[1] > y:
                    table_cell_number_scale[table_cell_number_value_position.index(item)] = to_scale
        
        
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table_cell_type_ids[i][j] == 2:
                if '%' in table[i][j]:
                    table_cell_number_scale[table_cell_number_value_position.index((i, j))] = 4
    
    return table_cell_number_scale
            
        
            
            
    


def table_tokenize(table, tokenizer, question_id):
    table_pd = pd.DataFrame(table, dtype=np.str)
    table_cell_type_ids = np.full(table_pd.shape, -1)
    table_is_scale_type = np.zeros(table_pd.shape)
    table_cell_tokens = []
    table_cell_number_value = []
    table_cell_number_value_position = []
    table_cell_number_value_position_b = []
    table_ids = []
    table_cell_index = []
    table_subword_lens = []
    table_types = []
    table_word_types = []
    current_cell_index = 0
    table_word_tokens = []
    table_word_tokens_number = []
    table_cell_types_flatten = []
    
    row_dict = defaultdict(list)
    column_dict = defaultdict(list)
    
    for i in range(len(table)):
        for j in range(len(table[i])):
            
            #一些特殊的处理
            table[i][j] = table[i][j].replace('bps', '')

            if table[i][j] in ['$—', '—%', '$ —', '— %', "-","—","$-","$ -","-%","- %"]:
                table[i][j] = '0'
            
            if is_None(table[i][j]):
                table_cell_type_ids[i][j] = 0
            elif is_number(table[i][j]) and not is_year(table[i][j]):
                table_cell_type_ids[i][j] = 2
            elif is_year(table[i][j]):
                table_cell_type_ids[i][j] = 1
            else:
                flag1 = 0
                for item in table[i]:
                    if item != '':
                        flag1 += 1
                
                flag = 0
                for item in table[i]:
                    if is_number(item) and not is_year(item):
                        flag = 1
                if flag:
                    table_cell_type_ids[i][j] = 3
                flag = 0
                for item in [item[j] for item in table]:
                    if is_number(item) and not is_year(item):
                        flag = 1
                if flag and flag1 != 1:
                    table_cell_type_ids[i][j] = 4
                
            if is_scale(table[i][j]) and not is_number(table[i][j]):
                table_is_scale_type[i][j] = 1
            
            if table_cell_type_ids[i][j] == 0:
                continue  
            table_cell_input_token = table[i][j]
            
                
            table_cell_tokens.append(table[i][j])
            
            if table_cell_type_ids[i][j] == 2:
                per_flag = 0
                for c_idx in range(len([item[j] for item in table])):
                    if table_is_scale_type[c_idx][j] == 1 and scale_judge(table[c_idx][j])==4:
                        per_flag = 1
                for c_idx in range(len(table[i])):
                    if table_is_scale_type[i][c_idx] == 1 and scale_judge(table[i][c_idx])==4:
                        per_flag = 1
                num_number = to_number(table[i][j]) 
                if per_flag and '%' not in table[i][j]:
                    num_number  = round(float(num_number/100.0),4)
                table_cell_number_value.append(num_number)
                table_cell_input_token = table[i][j].replace(' ','').replace('$','')
                if '(' in table_cell_input_token and ')' in table_cell_input_token:
                    table_cell_input_token = table_cell_input_token.replace('(', '').replace(')', '')
                    table_cell_input_token = '-' + table_cell_input_token
                table_cell_number_value_position.append((i, j))
            else:
                table_cell_number_value.append(np.nan)
                table_cell_number_value_position.append((-1, -1))
            cell_ids, string_subword_lens, tokens= string_tokenizer(table_cell_input_token, tokenizer)
            table_word_tokens.extend(tokens)
            table_word_tokens_number += [len(tokens)]
            table_subword_lens += string_subword_lens
            
            if table_cell_type_ids[i][j] == 2:
                table_types += [2 for _ in range(len(cell_ids))]
                table_word_types += [2 for _ in range(len(string_subword_lens))]
            else:
                table_types += [3 for _ in range(len(cell_ids))]
                table_word_types += [3 for _ in range(len(string_subword_lens))]
                
            table_ids += cell_ids
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]

            
            row_dict[i].append(current_cell_index)
            column_dict[j].append(current_cell_index)
            table_cell_types_flatten.append(table_cell_type_ids[i][j])
            current_cell_index += 1

    return table_pd,table_cell_type_ids,table_is_scale_type,table_cell_tokens,table_cell_number_value,\
        table_ids,table_cell_index, table_subword_lens, table_types, table_word_types,table_word_tokens,table_word_tokens_number,table_cell_number_value_position,\
            row_dict, column_dict, table_cell_types_flatten


def paragraph_number_judge(tokens, idx, this_number, paragraphs):
    global_scale = 0
    
    for i, paragraph_text in paragraphs.items():
        if 'in thousand' in paragraph_text:
            global_scale = 1
        if 'in million' in paragraph_text:
            global_scale = 2
        if 'in billion' in paragraph_text:
            global_scale = 3

    
    flag = 1
    scale_flag = 0
    if idx != 0:
        if is_month(tokens[idx-1]) and '$' not in this_number:
            flag = 0
    if idx != len(tokens) - 1:
        if is_month(tokens[idx+1]) and '$' not in this_number:
            flag = 0 
    if idx == 0:
        flag = 0


    if this_number[-1] == '.' and '$' not in this_number and '%' not in this_number and '£' not in this_number:
        flag = 0
                
    
    for stopword in ['(1)','(2)','(3)','(4)','(5)','(6)','(7)']:
        if stopword in this_number:
            flag = 0
            break
    
    if flag != 0:
        if '%' in this_number:
            scale_flag = 4
            
        if 'm' in this_number:
            scale_flag = 2
        if 'b' in this_number:
            scale_flag = 3
        
        for token in tokens[idx+1:idx+3]:
            token = sep_lower(token)
            for scale_idx, scale_label in enumerate(['dfe526f2', 'thousand', 'million', 'billion', 'percent']):
                if scale_label in token:
                    scale_flag = scale_idx
    
    if scale_flag == 0 and global_scale != 0:
        scale_flag = global_scale
    
    return flag, scale_flag
        
    
def paragraph_tokenize(question, paragraphs, tokenizer, question_id):
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []

    number_mask = []
    number_value = []

    tokens = []

    paragraph_subword_tags = []
    paragraph_index = []
    paragraph_tags = []
    
    paragraph_subword_lens = []
    paragraph_types = []
    paragraph_word_types = []
    paragraph_number_scale = []
    paragraph_sep_tag = []
    paragraph_mapping = False
    paragraph_mapping_orders = []

    sorted_order = get_order_by_tf_idf(question, paragraphs)

    last_token_length = 0
    for order in sorted_order:
        text = paragraphs[order]
        text = " ".join(text.split())
        prev_is_whitespace = True
        answer_indexs = None

        start_index = 0
        wait_add = False

        tokens.append(tokenizer.sep_token)

        for i, c in enumerate(text):
            if is_whitespace(c):
                if wait_add:
                    wait_add = False
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
    if "direct‐to‐grower" in tokens:
        tokens.remove("direct‐to‐grower")
    current_token_index = 0
    for i, token in enumerate(tokens):
        if token == tokenizer.sep_token:
            sub_tokens = tokenizer.tokenize(token)
            paragraph_sep_tag += [1]
        elif i != 0:
            sub_tokens = tokenizer.tokenize(" " + token)
            paragraph_sep_tag += [0]
        else:
            sub_tokens = tokenizer.tokenize(token)
            paragraph_sep_tag += [0]
        paragraph_subword_lens += [len(sub_tokens)]
        stop_flag = 0
        if is_number(token) and not is_year(token):
            stop_flag, scale_flag = paragraph_number_judge(tokens, i, token, paragraphs)
        if stop_flag == 1 and is_number(token) and not is_year(token):
            number_value.append(float(to_number(token)))
            paragraph_types += [2 for _ in range(len(sub_tokens))]
            paragraph_word_types += [2]
            paragraph_number_scale.append(scale_flag)
        else:
            number_value.append(np.nan)
            paragraph_types += [3 for _ in range(len(sub_tokens))]
            paragraph_word_types += [3]
            paragraph_number_scale.append(np.nan)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)

        current_token_index+=1
        paragraph_subword_tags += [1]
        if len(sub_tokens) > 1:
            paragraph_subword_tags += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)

    return tokens, paragraph_ids, number_value, paragraph_index,paragraph_subword_lens,paragraph_types,paragraph_word_types, paragraph_number_scale, paragraph_sep_tag
    
    
def subword_judge(subword_lens, input_types):
    aa = len(input_types) - len(torch.where(torch.tensor(input_types)==0)[0])
    bb = 0
    for i in subword_lens:
        bb += i
    if aa == bb:
        return subword_lens
    else:
        cc = bb - aa
        subword_lens[-1] -= cc
        return subword_lens
    

def _concat(question_ids,
            table_ids, 
            paragraph_ids, 
            question_tokens,
            table_tokens,
            paragraph_tokens,
            table_cell_number_value,
            paragraph_number_value,
            table_cell_index,
            paragraph_index,
            question_subword_lens,
            question_types,
            table_subword_lens,
            table_types,
            paragraph_subword_lens,
            paragraph_types,
            question_word_types,
            table_word_types,
            paragraph_word_types,
            table_cell_number_value_position,
            table_cell_number_scale,
            paragraph_number_scale,
            question_number_value,
            row_dict,
            column_dict,
            table_cell_types_flatten,
            sep,
            question_length_limitation,
            passage_length_limitation,
            max_pieces):
    
    input_ids = torch.zeros([1,max_pieces])
    token_type_ids = torch.zeros_like(input_ids)

    question_ids = [sep] + question_ids + [sep]
    question_length = len(question_ids)
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    table_cell_count_origin = len(table_cell_types_flatten)
    if passage_length_limitation is not None:
        passage_length_limitation = passage_length_limitation - question_length - 2
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
            table_cell_index = table_cell_index[:passage_length_limitation]
            table_types = table_types[:passage_length_limitation]
            end = table_cell_index[-1]
            table_subword_lens = table_subword_lens[:end+1]
            table_word_types = table_word_types[:end+1]
            table_tokens = table_tokens[:end+1]
            table_cell_types_flatten = table_cell_types_flatten[:end+1]
            table_cell_number_value = table_cell_number_value[:end+1]
            table_cell_number_value_position = table_cell_number_value_position[:end+1]
            table_cell_number_scale = table_cell_number_scale[:end+1]
        elif len(table_ids) + len(paragraph_ids) > passage_length_limitation:
            passage_ids = table_ids + [sep] + paragraph_ids
            passage_ids = passage_ids[:passage_length_limitation]
            paragraph_index=paragraph_index[:passage_length_limitation-table_length-1]
            paragraph_types = paragraph_types[:passage_length_limitation-table_length-1]
            end = paragraph_index[-1]
            paragraph_tokens = paragraph_tokens[:end+1]
            paragraph_number_value = paragraph_number_value[:end+1]
            paragraph_number_scale = paragraph_number_scale[:end+1]
            paragraph_subword_lens = paragraph_subword_lens[:end+1]
            paragraph_word_types = paragraph_word_types[:end+1]
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length
        else:
            passage_ids = table_ids + [sep] + paragraph_ids
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids) + 1
    else:
        passage_ids = table_ids + [sep] + paragraph_ids
    
    passage_ids = passage_ids + [sep]
    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    attention_mask = input_ids != 0

    input_types = question_types + table_types
    if paragraph_length > 1:
        input_types += [0] + paragraph_types
    
    
    try:
        table_number = [item for item in table_cell_number_value if not np.isnan(item)]
    except:
        import pdb; pdb.set_trace()
    
    paragraph_number = [item for item in paragraph_number_value if not np.isnan(item)]
    question_number = [item for item in question_number_value if not np.isnan(item)]
    table_number_position = [x for x, y in zip(table_cell_number_value_position, table_cell_number_value) if not np.isnan(y)]
    
    table_number_index = [i for i,item in enumerate(table_cell_number_value) if not np.isnan(item)]
    paragraph_number_index = [i for i,item in enumerate(paragraph_number_value) if not np.isnan(item)]
    question_number_index = [i for i,item in enumerate(question_number_value) if not np.isnan(item)]

    table_number_scale = [item for i,item in enumerate(table_cell_number_scale) if not np.isnan(item)]
    paragraph_number_scale = [item for i,item in enumerate(paragraph_number_scale) if not np.isnan(item)]
    
    subword_lens = question_subword_lens + table_subword_lens + paragraph_subword_lens
    input_word_types = question_word_types + table_word_types + paragraph_word_types
    
    
    subword_lens = subword_judge(subword_lens, input_types)
    
    table_cell_count = len(table_cell_types_flatten)
    if table_cell_count_origin != table_cell_count:
        import pdb; pdb.set_trace()
        
    for key, key_list in row_dict.items():
        for item in key_list:
            if item >= table_cell_count:
                key_list.remove(item)
    for key, key_list in column_dict.items():
        for item in key_list:
            if item >= table_cell_count:
                key_list.remove(item)
    
    
    return input_ids, attention_mask, input_types, input_word_types, subword_lens,\
        table_cell_index,table_tokens,table_number, table_number_index, paragraph_index,\
            paragraph_tokens, paragraph_number, paragraph_number_index,token_type_ids,table_number_position,\
                table_number_scale, paragraph_number_scale, question_number, question_number_index, row_dict, column_dict, table_cell_types_flatten



OPERATOR = ['+', '-', '*', '/']

def get_operators(derivation:str):
    res = []
    for c in derivation:
        if c in OPERATOR:
            res.append(c)
    return res

def delete_punc(example):
    if example == '[SEP]' or example == '</s>' or example == '<s>':
        return example
    punctuation_string = string.punctuation
    for i in punctuation_string:
        example = example.replace(i, '')
    return example
def sep_lower(example):
    if example == '[SEP]' or example == '</s>' or example == '<s>':
        return example
    else:
        return example.lower()


class GraphFactory(object):
    def __init__(self, question_tokens, table_tokens, paragraph_tokens, row_dict, column_dict, table_types, table, table_origin_types, tokenizer
        ):
        self.question_tokens = question_tokens
        self.table_tokens = table_tokens
        self.paragraph_tokens = paragraph_tokens
        self.global_tokens = question_tokens + table_tokens + paragraph_tokens
        self.table_types = table_types
        self.row_dict = row_dict
        self.column_dict = column_dict
        self.row_dict = OrderedDict(sorted(self.row_dict.items(), key=lambda x: x[0], reverse=False))
        self.column_dict = OrderedDict(sorted(self.column_dict.items(), key=lambda x: x[0], reverse=False))
        self.graph = defaultdict(list)
        self.q_n = len(question_tokens)
        self.t_n = len(table_tokens)
        self.p_n = len(paragraph_tokens)
        self.table = table
        self.table_origin_types = table_origin_types
        self.tokenizer = tokenizer
    
    def generate_graph(self):
        inner_table_relations = self.generate_inner_table_relation()
        inner_paragraph_relations = self.generate_inner_paragraph_relation()
        inner_question_relations = self.generate_inner_question_relation()
        question_table_relations = self.generate_question_table_relation()
        question_paragraph_relations = self.generate_question_paragraph_relation()
        paragraph_table_relations = self.generate_paragraph_table_relation()
        global_relations = inner_table_relations+inner_paragraph_relations+\
            inner_question_relations+question_table_relations+question_paragraph_relations+\
                paragraph_table_relations
        global_relations = sorted(global_relations, key=lambda x:(x[0],x[1]))
        
        global_relations_new = []
        punctuation_string = string.punctuation
        for relation in global_relations:
            string1 = self.global_tokens[relation[0]]
            string2 = self.global_tokens[relation[1]]
            
            if string1 in punctuation_string or string2 in punctuation_string:
                if not ((relation[0] > self.q_n+self.t_n \
                and relation[1] > self.q_n+self.t_n)
                or (relation[0] <= self.q_n \
                and relation[1] <= self.q_n)\
                or (relation[0] <= self.q_n+self.t_n and relation[0]>self.q_n  \
                and relation[1] <= self.q_n+self.t_n and relation[0]>self.q_n )
                    ):
                    continue
        
            if relation[0]>self.q_n+self.t_n+self.p_n or \
                relation[1]>self.q_n+self.t_n+self.p_n:
                continue
            global_relations_new.append(relation)
        self.relations = sorted(global_relations_new, key=lambda x:(x[0],x[1]))
        
        
        target_set = set(list(range(0, self.q_n+self.t_n+self.p_n)))
        current_set = set()
        for relation in self.relations:
            current_set.add(relation[0])
        cha = list(target_set-current_set)
        if cha:
            for chas in cha:
                self.relations.append((chas, chas, "self-same-self"))
        relation_set = list(set(self.relations))
        if len(relation_set) != len(self.relations):
            import pdb; pdb.set_trace()        
        
        graph = {}
        graph['src'] = list(map(lambda x:x[0], self.relations))
        graph['dst'] = list(map(lambda x:x[1], self.relations))
        graph['relations'] = list(map(lambda x:Relation_vocabulary['word2id'][x[2]], self.relations))
        graph['triplet'] = self.relations
        graph['dgl'] = dgl.graph((graph['src'], graph['dst']))
    
        return graph
        
        
    def type_check(self, idx):
        return self.table_types[idx]
    
    # 日期 1， row_head 3  column_head 4, number 2
    def generate_inner_table_relation(self):
        self.inner_table_relation = []
        
        for Row, R_list in self.row_dict.items():
            for i in range(len(R_list)):
                for j in range(len(R_list)):
                    if i == j:
                        continue
                    if self.type_check(R_list[i]) == 3 and self.type_check(R_list[j]) == 2:
                        self.inner_table_relation.append((self.q_n+R_list[i], self.q_n+R_list[j], "Row-Contain-Cell"))
                    elif self.type_check(R_list[i]) == 2 and self.type_check(R_list[j]) == 3:
                        self.inner_table_relation.append((self.q_n+R_list[i], self.q_n+R_list[j], "Cell-Belong-Row"))
                    elif self.type_check(R_list[i]) == 1 and self.type_check(R_list[j]) == 2:
                        self.inner_table_relation.append((self.q_n+R_list[i], self.q_n+R_list[j], "Time-Contain-Cell"))
                    elif self.type_check(R_list[i]) == 2 and self.type_check(R_list[j]) == 1:
                        self.inner_table_relation.append((self.q_n+R_list[i], self.q_n+R_list[j], "Cell-Belong-Time"))
                    elif self.type_check(R_list[i]) == 2 and self.type_check(R_list[j]) == 2:
                        self.inner_table_relation.append((self.q_n+R_list[i], self.q_n+R_list[j], "Cell-SameRow-Cell"))
        for Column, C_list in self.column_dict.items():
            for i in range(len(C_list)):
                for j in range(len(C_list)):
                    if i == j:
                        continue
                    if self.type_check(C_list[i]) == 4 and self.type_check(C_list[j]) == 2:
                        self.inner_table_relation.append((self.q_n+C_list[i], self.q_n+C_list[j], "Column-Contain-Cell"))
                    elif self.type_check(C_list[i]) == 2 and self.type_check(C_list[j]) == 4:
                        self.inner_table_relation.append((self.q_n+C_list[i], self.q_n+C_list[j], "Cell-Belong-Column"))
                    elif self.type_check(C_list[i]) == 1 and self.type_check(C_list[j]) == 2:
                        self.inner_table_relation.append((self.q_n+C_list[i], self.q_n+C_list[j], "Time-Contain-Cell"))
                    elif self.type_check(C_list[i]) == 2 and self.type_check(C_list[j]) == 1:
                        self.inner_table_relation.append((self.q_n+C_list[i], self.q_n+C_list[j], "Cell-Belong-Time"))

        xx, yy = self.table_origin_types.shape
        special_year = 0
        for i, row in enumerate(self.table_origin_types):
            if row[0] == 1 and np.sum(row) == 1 and i != 0:
                special_year += 1
        if special_year > 1:
            table_types_numpy = np.array(self.table_types)
            year_tags = np.where(table_types_numpy==1)[0]
            for i in range(len(year_tags)-1):
                start = year_tags[i]
                for j in range(year_tags[i]+1, year_tags[i+1]):
                    if self.table_types[j] == 2:
                        if (self.q_n+start, self.q_n+j, "Time-Contain-Cell") not in self.inner_table_relation:
                            self.inner_table_relation.append((self.q_n+start, self.q_n+j, "Time-Contain-Cell"))
                        if (self.q_n+j, self.q_n+start, "Cell-Belong-Time") not in self.inner_table_relation:
                            self.inner_table_relation.append((self.q_n+j, self.q_n+start, "Cell-Belong-Time"))
            start = year_tags[-1]
            for j in range(year_tags[-1]+1, len(self.table_types)):
                if self.table_types[j] == 2:
                    if (self.q_n+start, self.q_n+j, "Time-Contain-Cell") not in self.inner_table_relation:
                        self.inner_table_relation.append((self.q_n+start, self.q_n+j, "Time-Contain-Cell"))
                    if (self.q_n+j, self.q_n+start, "Cell-Belong-Time") not in self.inner_table_relation:
                        self.inner_table_relation.append((self.q_n+j, self.q_n+start, "Cell-Belong-Time"))

        self.inner_table_relation = sorted(self.inner_table_relation, key=lambda x:(x[0],x[1]))

        return self.inner_table_relation
    
    def generate_inner_paragraph_relation(self):
        self.inner_paragraph_relation = []
        select_node = []
        for idx, node in enumerate(self.paragraph_tokens):
            if node == self.tokenizer.sep_token:
                select_node.append(idx)
        paragraph_split = []
        last_select_node = 0
        for idx, node in enumerate(self.paragraph_tokens):
            if node == self.tokenizer.sep_token:
                last_select_node = idx
            if node != self.tokenizer.sep_token:
                self.inner_paragraph_relation.append((self.q_n+self.t_n+last_select_node, self.q_n+self.t_n+idx, "Sentence-Contain-P_Word"))
                self.inner_paragraph_relation.append((self.q_n+self.t_n+idx, self.q_n+self.t_n+last_select_node, "P_Word-Partof-Sentence"))
        for i in range(len(self.paragraph_tokens) - 1):
            if self.paragraph_tokens[i] != self.tokenizer.sep_token and self.paragraph_tokens[i+1] != self.tokenizer.sep_token:
                self.inner_paragraph_relation.append((self.q_n+self.t_n+i, self.q_n+self.t_n+i+1, "Q_Word-PriorWord-Q_Word"))
                self.inner_paragraph_relation.append((self.q_n+self.t_n+i+1, self.q_n+self.t_n+i, "Q_Word-NextWord-Q_Word"))
        self.inner_paragraph_relation = sorted(self.inner_paragraph_relation, key=lambda x:(x[0],x[1]))
        return self.inner_paragraph_relation
    
    def generate_inner_question_relation(self):
        self.inner_question_relations = []
        for i in range(len(self.question_tokens) - 1):
            self.inner_question_relations.append((i, i+1, "Q_Word-PriorWord-Q_Word"))
            self.inner_question_relations.append((i+1, i, "Q_Word-NextWord-Q_Word"))
        self.inner_question_relations = sorted(self.inner_question_relations, key=lambda x:(x[0],x[1]))
        return self.inner_question_relations

    def generate_question_table_relation(self):
        question = " ".join(self.question_tokens)
        question = question.lower()
        self.question_table_relations = []
        
        question_tokens = list(map(delete_punc, self.question_tokens))
        question_tokens = list(map(sep_lower, question_tokens))
        table_tokens = list(map(delete_punc, self.table_tokens))
        table_tokens = list(map(sep_lower, table_tokens))
        
        for i, word in enumerate(question_tokens):
            word = sep_lower(word)
            if word in stop_words:
                continue
            for j, cell in enumerate(table_tokens):
                flag1, flag2 = 0, 0
                cell = cell.strip(':')
                cell = sep_lower(cell)
                cell_words = cell.split(' ')
                if self.table_types[j] == 2:
                    continue
                if word in cell_words:
                    flag1 = 1
                if cell in question:
                    flag2 = 1
                if flag1 and flag2:
                    self.question_table_relations.append((i, self.q_n+j, "Q_Word-ExactMatch-Row"))
                    self.question_table_relations.append((self.q_n+j, i, "Row-ExactMatch-Q_Word"))
                elif flag1 and not flag2:
                    self.question_table_relations.append((i, self.q_n+j, "Q_Word-PartialMatch-Row"))
                    self.question_table_relations.append((j+self.q_n, i, "Row-PartialMatch-Q_Word"))
        self.question_table_relations = sorted(self.question_table_relations, key=lambda x:(x[0],x[1]))

        return self.question_table_relations
                
    def generate_question_paragraph_relation(self):
        self.question_paragraph_relations = []
        question_tokens = list(map(delete_punc, self.question_tokens))
        question_tokens = list(map(sep_lower, question_tokens))
        paragraph_tokens = list(map(delete_punc, self.paragraph_tokens))
        paragraph_tokens = list(map(sep_lower, paragraph_tokens))
        
        select_node = []
        for idx, node in enumerate(paragraph_tokens):
            if node == self.tokenizer.sep_token:
                select_node.append(idx)
        split_number = len(select_node)
        
        paragraph_split = [list(g) for k, g in groupby(paragraph_tokens, lambda x:x==self.tokenizer.sep_token) if not k]

        
        for i, paragraph in enumerate(paragraph_split):
            for j, word in enumerate(question_tokens):
                if word in stop_words or word == '':
                    continue
                if word in paragraph:
                    self.question_paragraph_relations.append((j, select_node[i]+self.q_n+self.t_n, "Q_Word-Partof-Sentence"))
                    self.question_paragraph_relations.append((select_node[i]+self.q_n+self.t_n, j, "Sentence-Contain-Q_Word"))
        
        for i, p_word in enumerate(paragraph_tokens):
            if p_word in stop_words or p_word == '':
                continue
            for j, word in enumerate(question_tokens):
                if p_word == word:
                    self.question_paragraph_relations.append((j, i+self.q_n+self.t_n, "Word-Same-Word"))
                    self.question_paragraph_relations.append((i+self.q_n+self.t_n, j, "Word-Same-Word"))
        self.question_paragraph_relations = sorted(self.question_paragraph_relations, key=lambda x:(x[0],x[1]))
        return self.question_paragraph_relations
    
    def generate_paragraph_table_relation(self):
        self.paragraph_table_relations = []
        table_tokens = list(map(delete_punc, self.table_tokens))
        table_tokens = list(map(sep_lower, table_tokens))
        paragraph_tokens = list(map(delete_punc, self.paragraph_tokens))
        paragraph_tokens = list(map(sep_lower, paragraph_tokens))
        
        select_node = []
        for idx, node in enumerate(paragraph_tokens):
            if node == self.tokenizer.sep_token:
                select_node.append(idx)
        split_number = len(select_node)
        
        paragraph_split = [list(g) for k, g in groupby(paragraph_tokens, lambda x:x==self.tokenizer.sep_token) if not k]
        
        for id, paragraph_tokens_new in enumerate(paragraph_split):
            paragraph_tokens_new_new = paragraph_tokens_new.copy()
            a = paragraph_tokens_new_new.count('')
            for _ in range(a):
                paragraph_tokens_new_new.remove('')
            paragraph = " ".join(paragraph_tokens_new_new)
            for i, word in enumerate(paragraph_tokens_new):
                word = sep_lower(word)
                if word in stop_words:
                    continue
                for j, cell in enumerate(table_tokens):
                    flag1, flag2 = 0, 0
                    cell = cell.strip(':')
                    cell = sep_lower(cell)
                    cell_words = cell.split(' ')
                    if self.table_types[j] == 2:
                        continue
                    if word in cell_words:
                        flag1 = 1
                    if cell in paragraph:
                        flag2 = 1
                    if flag1 and flag2:
                        self.paragraph_table_relations.append((select_node[id]+i+1+self.q_n+self.t_n, j+self.q_n, "P_Word-ExactMatch-Row"))
                        self.paragraph_table_relations.append((j+self.q_n, select_node[id]+i+1+self.q_n+self.t_n, "Row-ExactMatch-P_Word"))
                    elif flag1 and not flag2:
                        self.paragraph_table_relations.append((select_node[id]+i+1+self.q_n+self.t_n, j+self.q_n, "P_Word-PartialMatch-Row"))
                        self.paragraph_table_relations.append((j+self.q_n, select_node[id]+i+1+self.q_n+self.t_n, "Row-PartialMatch-P_Word"))
            for k, cell2 in enumerate(table_tokens):
                cell2 = cell2.strip(':')
                cell2 = sep_lower(cell2)
                cell_words = cell2.split(' ')
                if self.table_types[k] == 2:
                    continue
                if cell2 in paragraph:
                    self.paragraph_table_relations.append((select_node[id]+self.q_n+self.t_n, k+self.q_n, "Sentence-Contain-Row"))
                    self.paragraph_table_relations.append((k+self.q_n, select_node[id]+self.q_n+self.t_n, "Row-Partof-Sentence"))
        self.paragraph_table_relations = sorted(self.paragraph_table_relations, key=lambda x:(x[0],x[1]))             
        return self.paragraph_table_relations
        



class TaTQATestReader(object):
    def __init__(self, tokenizer, 
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>"):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep = self.tokenizer.convert_tokens_to_ids(sep)
        self.skip_count = 0
    
    def _make_instance(self,input_ids, attention_mask,token_type_ids,
                question_tokens,
                table, table_cell_type_ids,table_is_scale_type, table_number, table_number_index, 
                table_cell_index, table_tokens,paragraph_tokens,
                paragraph_index,paragraph_number,paragraph_number_index,
                question_id,
                word_mask, number_mask, subword_lens, item_length,
                word_word_mask, number_word_mask, table_word_tokens, table_word_tokens_number,
                table_number_scale, paragraph_number_scale, graph, question, question_number,question_number_index):
        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),

            "question_tokens": question_tokens,

            "table": table,
            "table_cell_type_ids": table_cell_type_ids,
            "table_is_scale_type": table_is_scale_type,
            "table_number": table_number,
            "table_number_index": np.array(table_number_index),
            "table_cell_index": np.array(table_cell_index),
            "table_tokens": table_tokens,

            "paragraph_tokens": paragraph_tokens,
            "paragraph_index": np.array(paragraph_index),
            "paragraph_number": paragraph_number,
            "paragraph_number_index": np.array(paragraph_number_index),

            "question_id": question_id,
            "word_mask": np.array(word_mask),
            "number_mask": np.array(number_mask),
            "subword_lens": np.array(subword_lens),
            "item_length": item_length,
            
            "word_word_mask": np.array(word_word_mask),
            "number_word_mask": np.array(number_word_mask),
            
            "table_word_tokens": table_word_tokens,
            
            "table_word_tokens_number": table_word_tokens_number,
            
            "table_number_scale": table_number_scale,
            "paragraph_number_scale" :paragraph_number_scale,
            
            "graph": graph,
            
            "question": question,

            "question_number": question_number,
            "question_number_index": question_number_index
        }        
    
    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], question_id:str):
        
        table_pd,table_cell_type_ids,table_is_scale_type,table_cell_tokens,\
        table_cell_number_value,table_ids,\
        table_cell_index, table_subword_lens, table_types, table_word_types,\
        table_word_tokens, table_word_tokens_number, table_cell_number_value_position,\
        row_dict, column_dict, table_cell_types_flatten = table_tokenize(table, self.tokenizer,question_id)
        

        
        paragraph_tokens, paragraph_ids, number_value, \
        paragraph_index, paragraph_subword_lens,paragraph_types, paragraph_word_types, paragraph_number_scale, paragraph_sep_tag\
        = paragraph_tokenize(question, paragraphs, self.tokenizer, question_id)
        
        question_ids, question_tokens, question_subword_lens, question_word_index, question_types, question_word_types, question_number_value\
        = question_tokenize(question, self.tokenizer)
        
        table_cell_number_scale = generate_number_scale(table, table_pd, table_cell_type_ids, table_is_scale_type, table_cell_number_value, table_cell_number_value_position, paragraphs)


        input_ids, attention_mask, input_types, input_word_types, subword_lens, \
        table_cell_index,table_tokens,table_number, table_number_index, paragraph_index,\
        paragraph_tokens, paragraph_number, paragraph_number_index,token_type_ids, \
        table_number_position,table_number_scale, paragraph_number_scale,question_number, question_number_index,\
        row_dict, column_dict, table_cell_types_flatten\
        = _concat(question_ids, table_ids, paragraph_ids,\
        question_tokens,table_cell_tokens, paragraph_tokens,table_cell_number_value,number_value,\
        table_cell_index, paragraph_index,question_subword_lens,question_types,table_subword_lens,\
        table_types, paragraph_subword_lens,paragraph_types,question_word_types,table_word_types,\
        paragraph_word_types,table_cell_number_value_position,table_cell_number_scale, paragraph_number_scale, question_number_value, \
        row_dict, column_dict, table_cell_types_flatten,
        self.sep,self.question_length_limit,self.passage_length_limit, self.max_pieces)
        
        
        item_length = {'q': len(question_tokens), 't': len(table_tokens), 'p':len(paragraph_tokens)}
        
        word_mask = torch.zeros(512)
        number_mask = torch.zeros(512)
        
        try:
            for i, idx in enumerate(input_types):
                if idx == 3:
                    word_mask[i] = 1
                if idx == 2:
                    number_mask[i] = 1
        except:
            word_mask = torch.zeros(512)
            number_mask = torch.zeros(512)
        
        word_word_mask = []
        number_word_mask = []
        for i, idx in enumerate(input_word_types):
            if idx == 3:
                word_word_mask += [1]
                number_word_mask += [0]
            if idx == 2:
                word_word_mask += [0]
                number_word_mask += [1]
                
        graphFactory = GraphFactory(question_tokens, table_tokens, paragraph_tokens, row_dict, column_dict, table_cell_types_flatten, table_pd, table_cell_type_ids, self.tokenizer)
        graph = graphFactory.generate_graph()
        
        return self._make_instance(input_ids, attention_mask,token_type_ids,
                question_tokens,
                table_pd, table_cell_type_ids,table_is_scale_type, table_number, table_number_index, table_cell_index, table_tokens,
                paragraph_tokens,paragraph_index,paragraph_number,paragraph_number_index, question_id,
                word_mask, number_mask, subword_lens, item_length, word_word_mask, number_word_mask,table_word_tokens,
                table_word_tokens_number, table_number_scale, paragraph_number_scale, graph, question,question_number, question_number_index)
        
    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances = []
        count = 0
        total = 0
        total_count = defaultdict(int)
        for one in tqdm(dataset):
            table = one['table']['table']
            if one['table']['uid'] == '097f22c33fd3ff811a21c799dd76e595':
                table = table[:-2]
            paragraphs = one['paragraphs']
            questions = one['questions']
            
            for question_answer in questions:
                question = question_answer["question"].strip()
                question_id = question_answer["uid"]
                instance = self._to_instance(question, table, paragraphs,question_id)
                if instance is not None:
                    instances.append(instance)
        print(len(instances))

        return instances
    
