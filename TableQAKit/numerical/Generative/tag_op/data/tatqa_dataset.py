import re
import string
import json
from tqdm import tqdm
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from typing import List, Dict, Tuple
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tag_op.data.file_utils import is_scatter_available
from tatqa_utils import  *
from tag_op.data.data_util import *
from tag_op.data.data_util import  _is_average, _is_change_ratio, _is_diff, _is_division, _is_sum, _is_times
from tag_op.data.operation_util import LogicalFormer, GRAMMER_CLASS, GRAMMER_ID

USTRIPPED_CHARACTERS = ''.join([u"Ġ"])
# soft dependency
if is_scatter_available():
    from torch_scatter import scatter

def convert_start_end_tags(split_tags, paragraph_index):
    in_split_tags = split_tags.copy()
    split_tags = [0 for i in range(len(split_tags))]
    for i in range(len(in_split_tags)):
        if in_split_tags[i] == 1:
            current_index = paragraph_index[i]
            split_tags[i] = 1
            paragraph_index_ = paragraph_index[i:]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[j] == current_index:
                    split_tags[i+j] = 1
                else:
                    break
            break
    for i in range(1, len(in_split_tags)):
        if in_split_tags[-i] == 1:
            current_index = paragraph_index[-i]
            split_tags[-i] = 1
            paragraph_index_ = paragraph_index[:-i]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[-j] == current_index:
                    split_tags[-i-j] = 1
                else:
                    break
            break
    del in_split_tags
    return split_tags

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def sortFunc(elem):
    return elem[1]

def get_order_by_tf_idf(question, paragraphs):
    sorted_order = []
    corpus = [question]
    for order, text in paragraphs.items():
        corpus.append(text)
        sorted_order.append(order)
    import pdb
    
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x:x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]


def get_answer_nums(table_answer_coordinates: List, paragraph_answer_coordinates: Dict):
    if table_answer_coordinates is not None:
        table_answer_num = len(table_answer_coordinates)
    else:
        table_answer_num = 0
    paragraph_answer_nums = 0
    if paragraph_answer_coordinates:
        for value in paragraph_answer_coordinates.values():
            paragraph_answer_nums += len(value)
    return table_answer_num, paragraph_answer_nums

def get_operands_index(label_ids, token_type_ids):
    row_ids = token_type_ids[:, :, 2]
    column_ids = token_type_ids[:, :, 1]
    max_num_rows = 64
    max_num_columns = 32
    row_index = IndexMap(
        indices=torch.min(row_ids, torch.as_tensor(max_num_rows - 1, device=row_ids.device)),
        num_segments=max_num_rows,
        batch_dims=1,
    )
    col_index = IndexMap(
        indices=torch.min(column_ids, torch.as_tensor(max_num_columns - 1, device=column_ids.device)),
        num_segments=max_num_columns,
        batch_dims=1,
    )
    cell_index = ProductIndexMap(row_index, col_index).indices
    first_operand_start = torch.argmax((label_ids!=0).int(), dim=1)[0]
    label_ids = label_ids[0, first_operand_start:]
    cell_index_first = cell_index[0, first_operand_start:]
    first_operand_end = torch.argmax(((cell_index_first-cell_index[0, first_operand_start])!=0).int())

    label_ids = label_ids[first_operand_end:]
    cell_index_first = cell_index_first[first_operand_end:]
    first_operand_end = first_operand_end+first_operand_start

    second_operand_start = torch.argmax((label_ids!=0).int())
    cell_index_second = cell_index_first[second_operand_start:]
    second_operand_end = torch.argmax(((cell_index_second-cell_index_first[second_operand_start])!=0).int())+second_operand_start
    second_operand_start+=first_operand_end
    second_operand_end+=first_operand_end
    return first_operand_start, first_operand_end, second_operand_start, second_operand_end

def get_tokens_from_ids(ids, tokenizer):
    tokens = []
    sub_tokens = []
    for id in ids:
        token = tokenizer._convert_id_to_token(id)
        if len(sub_tokens) == 0:
            sub_tokens.append(token)
        elif str(token).startswith("##"):
            sub_tokens.append(token[2:])
        elif len(sub_tokens) != 0:
            tokens.append("".join(sub_tokens))
            sub_tokens = [token]
    tokens.append("".join(sub_tokens))
    return "".join(tokens)

def get_number_mask(table):
    max_num_rows = 64
    max_num_columns = 32
    columns = table.columns.tolist()
    number_mask = np.zeros((1, max_num_columns*max_num_rows))
    number_value = np.ones((1, max_num_columns*max_num_rows)) * np.nan
    for index, row in table.iterrows():
        for col_index in columns:
            col_index = int(col_index)
            in_cell_index = (index+1)*max_num_columns+col_index+1
            table_content = row[col_index]
            number = to_number(table_content)
            if number is not None:
                number_mask[0, in_cell_index] = 1
                number_value[0, in_cell_index] = float(number)
    return number_mask, number_value

def tokenize_answer(answer):
    answer_tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(answer):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            answer_tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                answer_tokens.append(c)
            else:
                answer_tokens[-1] += c
            prev_is_whitespace = False
    return answer_tokens

def string_tokenizer(string: str, tokenizer) -> List[int]:
    if not string:
        return [], []
    tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(string):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
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
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return split_tokens, ids

def table_tokenize(table, tokenizer, mapping=None, answer_type=None):
    table_cell_tokens = []
    table_ids = []
    table_tags = []
    table_cell_index = []
    table_cell_number_value = []
    table_mapping = False
    answer_coordinates = None
    
    table_row_header_index = []
    table_col_header_index = []
    table_data_cell_index = []
    all_cell_index = []
    all_cell_token_index = []
    table_split_cell_tokens = []
    
    same_row_data_cell_rel = []
    same_col_data_cell_rel = []
    data_cell_row_rel = {}
    data_cell_col_rel = {}
    
    
    ## row col
    row_tags = []
    col_tags = [0]* len(table[0])
    row_include_cells = {}
    col_include_cells = {}
    
    
    
    if mapping is not None and "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]
    
    current_cell_index = 1
    row_id = 0
    cell_token_start_index = 0
    
    for i in range(len(table)):
        skip_row = True ## all the cell is None in this row
        current_row_cell = []
        for k in range(len(table[i])):
            split_tokens, cell_ids = string_tokenizer(table[i][k], tokenizer)
            if not cell_ids:
                continue
            skip_row=False
            table_ids += cell_ids
            table_split_cell_tokens += split_tokens
            
            ## add row and col header in table
            if i==0:
                table_col_header_index.append((current_cell_index,k))
            if k==0:
                table_row_header_index.append((current_cell_index,i))
            if i!=0 and k!=0:
                table_data_cell_index.append((current_cell_index, i, k))
                
            all_cell_index.append((current_cell_index, i, k))
            current_row_cell.append(current_cell_index)
            all_cell_token_index.append((cell_token_start_index, cell_token_start_index+ len(cell_ids)))
            cell_token_start_index+=len(cell_ids)
            
            
            if is_number(table[i][k]):
                ## % 预处理
                table_cell_copy = table[i][k]
                table_cell_copy = table_cell_copy.replace("%", "")
                table_cell_number_value.append(to_number(table_cell_copy))

            else:
                table_cell_number_value.append(np.nan)
            table_cell_tokens.append(table[i][k])
            
            if table_mapping:
                if [i, k] in answer_coordinates:
                    table_tags += [1 for _ in range(len(cell_ids))]
                    
                    ## add col tag
                    col_tags[k] = 1
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]
            current_cell_index += 1
        
        ## row tag
        if not skip_row: ## this row can't be skip since exsisting a cell is not none.
            if answer_coordinates and i in [item[0] for item in answer_coordinates]:
                row_tags.append(1)
            else:
                row_tags.append(0)
            row_include_cells.update({row_id:current_row_cell})
            row_id+=1
    
    ## col_tag
    for col_id in range(len(col_tags)):
        cells = list(filter(lambda x:x[2]==col_id,  all_cell_index))
        cell_ids = [item[0] for item in cells]
        col_include_cells.update({col_id:cell_ids})
    
#     import pdb
#     pdb.set_trace()
    
    
    ## add table structure knowledge
    for col_header in table_col_header_index:
        col_id = col_header[1]
        data_cell_in_current_col = list(filter(lambda x:x[2]==col_id,  table_data_cell_index))
        if len(data_cell_in_current_col)>0:
            data_cell_col_rel.update({col_header[0]:[item[0] for item in data_cell_in_current_col]})
    
    for row_header in table_row_header_index:
        row_id = row_header[1]
        data_cell_in_current_row = list(filter(lambda x:x[1]==row_id,  table_data_cell_index))
        if len(data_cell_in_current_row)>0:
            data_cell_row_rel.update({row_header[0]:[item[0] for item in data_cell_in_current_row]})
    
    all_row_id = []
    all_col_id = []
    
    for data_cell in table_data_cell_index:
        cell_id = data_cell[0]
        row_id = data_cell[1]
        col_id = data_cell[2]

        if row_id not in all_row_id:
            same_row_data_cell_rel.append([cell_id])
            all_row_id.append(row_id)
        else:
            index = all_row_id.index(row_id)
            same_row_data_cell_rel[index].append(cell_id)
        
            
        if col_id not in all_col_id:
            same_col_data_cell_rel.append([cell_id])
            all_col_id.append(col_id)
        else:
            index = all_col_id.index(col_id)
            same_col_data_cell_rel[index].append(cell_id)

            
    time_cell_indices = []
    for idx, value in enumerate(table_cell_number_value):
        flag = years_detector(str(value))
        if len(flag)>0:
            time_cell_indices.append(idx)
    ## add timestamp for each cell in table
    table_cell_number_value_with_time = get_times_tamp(time_cell_indices, all_cell_index, table_cell_number_value)
    
    
    ## add row including cell / col including cell
    return table_cell_tokens, table_split_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index,       table_row_header_index, table_col_header_index, table_data_cell_index,same_row_data_cell_rel, same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index
       

    


def table_test_tokenize(question_id, table, tokenizer, mapping, answer_type):
    mapping_content = []
    table_cell_tokens = []
    table_ids = []
    table_tags = []
    table_cell_index = []
    table_cell_number_value = []
    table_mapping = False
    answer_coordinates = None
    
    
    table_row_header_index = []
    table_col_header_index = []
    table_data_cell_index = []
    
    ## for order
    all_cell_index = []
    
    
    
    same_row_data_cell_rel = []
    same_col_data_cell_rel = []
    data_cell_row_rel = {}
    data_cell_col_rel = {}
    
    
    ## row col
    row_tags = []
    col_tags = [0]* len(table[0])
    row_include_cells = {}
    col_include_cells = {}
    
    

    if "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]
    
    
    
    current_cell_index = 1
    row_id = 0
    for i in range(len(table)):
        skip_row = True ## all the cell is None in this row
        current_row_cell = []
        for k in range(len(table[i])):
            split_tokens, cell_ids = string_tokenizer(table[i][k], tokenizer)
            if not cell_ids:
                continue
                
            skip_row=False
            table_ids += cell_ids
            if i==0:
                table_col_header_index.append((current_cell_index,k))
            if k==0:
                table_row_header_index.append((current_cell_index,i))
            if i!=0 and k!=0:
                table_data_cell_index.append((current_cell_index, i, k))
            
            all_cell_index.append((current_cell_index, i, k))
            current_row_cell.append(current_cell_index)
            
            if is_number(table[i][k]):
#                 table_cell_number_value.append(to_number(table[i][k]))
                 ## % 预处理
                table_cell_copy = table[i][k]
                table_cell_copy = table_cell_copy.replace("%", "")
                table_cell_number_value.append(to_number(table_cell_copy))
            else:
                table_cell_number_value.append(np.nan)
            table_cell_tokens.append(table[i][k])
            if table_mapping:
                if [i, k] in answer_coordinates:
                    mapping_content.append(table[i][k])
                    table_tags += [1 for _ in range(len(cell_ids))]
                    ## add col tag
                    col_tags[k] = 1
                    
                else:
                    table_tags += [0 for _ in range(len(cell_ids))]
            else:
                table_tags += [0 for _ in range(len(cell_ids))]
            table_cell_index += [current_cell_index for _ in range(len(cell_ids))]
            current_cell_index += 1
    
## row tag
        if not skip_row: ## this row can't be skip since exsisting a cell is not none.
            if answer_coordinates and i in [item[0] for item in answer_coordinates]:
                row_tags.append(1)
            else:
                row_tags.append(0)
            row_include_cells.update({row_id:current_row_cell})
            row_id+=1
    
    ## col_tag
    for col_id in range(len(col_tags)):
        cells = list(filter(lambda x:x[2]==col_id,  all_cell_index))
        cell_ids = [item[0] for item in cells]
        col_include_cells.update({col_id:cell_ids})
    
    ## add table structure knowledge
    for col_header in table_col_header_index:
        col_id = col_header[1]
        data_cell_in_current_col = list(filter(lambda x:x[2]==col_id,  table_data_cell_index))
        if len(data_cell_in_current_col)>0:
            data_cell_col_rel.update({col_header[0]:[item[0] for item in data_cell_in_current_col]})
    
    for row_header in table_row_header_index:
        row_id = row_header[1]
        data_cell_in_current_row = list(filter(lambda x:x[1]==row_id,  table_data_cell_index))
        if len(data_cell_in_current_row)>0:
            data_cell_row_rel.update({row_header[0]:[item[0] for item in data_cell_in_current_row]})
    
    all_row_id = []
    all_col_id = []
    
    for data_cell in table_data_cell_index:
        cell_id = data_cell[0]
        row_id = data_cell[1]
        col_id = data_cell[2]

        if row_id not in all_row_id:
            same_row_data_cell_rel.append([cell_id])
            all_row_id.append(row_id)
        else:
            index = all_row_id.index(row_id)
            same_row_data_cell_rel[index].append(cell_id)
        
            
        if col_id not in all_col_id:
            same_col_data_cell_rel.append([cell_id])
            all_col_id.append(col_id)
        else:
            index = all_col_id.index(col_id)
            same_col_data_cell_rel[index].append(cell_id)
            
    time_cell_indices = []
#     for idx, value in enumerate(table_cell_number_value):
#         flag = years_detector(str(value))
#         if len(flag)>0:
#             time_cell_indices.append(idx)
            
    for idx, value in enumerate(all_cell_index):
        i = value[1]
        k = value[2]
        flag = years_detector(table[i][k])
        if len(flag)>0:
            time_cell_indices.append(idx)
            table_cell_number_value[idx] = int(flag[0])
    

    ## add timestamp for each cell in table
    if question_id == "f3c2a0c3-eb77-4161-94e3-98ca2740e978":
        import pdb
        pdb.set_trace()
    table_cell_number_value_with_time = get_times_tamp(time_cell_indices, all_cell_index, table_cell_number_value)
    return table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index, table_row_header_index, table_col_header_index, table_data_cell_index, same_row_data_cell_rel, same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index



def get_times_tamp(time_cell_indices, all_cell_index, table_cell_number_value):
    time_cell_col = [all_cell_index[idx][2] for idx in time_cell_indices]
    time_cell_value = [table_cell_number_value[idx] for idx in time_cell_indices]
    
    table_cell_number_value_with_time = []
    for id, item in enumerate(all_cell_index):
        if item[2] not in time_cell_col:
            table_cell_number_value_with_time.append(np.nan)
            continue
        time_idx = time_cell_col.index(item[2])
        table_cell_number_value_with_time.append(time_cell_value[time_idx])
    assert len(table_cell_number_value_with_time) == len(table_cell_number_value)
    return table_cell_number_value_with_time


def paragraph_tokenize(question, paragraphs, tokenizer, mapping, answer_type):
    '''
    @@ parameters
    question: a string
    paragraphs: a dict of which key is id and values is the text of a paragraph
    tokenizer:
    mapping: ground truth index , if answer in table, index is row_index and col index; if anwer in paragraph, key is paragraph                id and value is offset
    answer_type: 
    @@return
    '''
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []
    split_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    word_piece_mask = []
    paragraph_index = []
    
    paragraph_mapping = False
    paragraph_mapping_orders = []
    if mapping is not None and "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]  ## answer offset
        current_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]

        start_index = 0
        wait_add = False ## if wait_add is true, the char is not the end of word
        ## tags of word splited by space and tokens obtained by splitting with space
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                import pdb
               
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                       
                        import pdb
#                         pdb.set_trace()
                    else:
                        tags.append(0)
#                         tokens.append(c)
#                         tags.append(0)
                    wait_add = False
                tokens.append(c)
                
                if len(tags)>0 and tags[-1]==1:
#                     pdb.set_trace()
                    tags.append(1)
                else:
                    tags.append(0)

#                 pdb.set_trace()
#                 tags.append(0)
                    
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)
                
    try:
        assert len(tokens) == len(tags)
    except AssertionError:
        print(len(tokens), len(tags))
        input()
    current_token_index = 1
    
    
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        number = to_number(token)
        if number is not None:
            number_value.append(float(number))
        else:
            number_value.append(np.nan)
        for sub_token in sub_tokens:
            split_tags.append(tags[i])
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)
#     if question=="Which model is used to estimate the fair value of employee share option?":
#         import pdb
#         pdb.set_trace()
    
    return tokens, split_tokens, paragraph_ids, split_tags, word_piece_mask, number_mask, number_value, paragraph_index

def paragraph_test_tokenize(question, paragraphs, tokenizer, mapping, answer_type):
    mapping_content = []
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tokens = []
    split_tags = []
    number_mask = []
    number_value = []
    tokens = []
    tags = []
    word_piece_mask = []
    paragraph_index = []

    paragraph_mapping = False
    paragraph_mapping_orders = []
    if "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        current_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                mapping_content.append(text[answer_index[0]:answer_index[1]])
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]
        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                tokens.append(c)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)
    try:
        assert len(tokens) == len(tags)
    except AssertionError:
        print(len(tokens), len(tags))
        input()
    current_token_index = 1
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        number = to_number(token)
        if number is not None:
            number_value.append(float(number))
        else:
            number_value.append(np.nan)
        for sub_token in sub_tokens:
            split_tags.append(tags[i])
            split_tokens.append(sub_token)
            paragraph_index.append(current_token_index)
        current_token_index+=1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    paragraph_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    
    return tokens, paragraph_ids, split_tags, word_piece_mask, number_mask, number_value, \
           paragraph_index, mapping_content


def question_tokenizer(question_text, tokenizer):
    return string_tokenizer(question_text, tokenizer)

def get_number_order_labels(paragraphs, table, derivation, operator_class, answer_mapping, question_id, OPERATOR_CLASSES):
    if ("DIVIDE" not in OPERATOR_CLASSES or operator_class != OPERATOR_CLASSES["DIVIDE"]) and \
            ("CHANGE_RATIO" not in OPERATOR_CLASSES or operator_class != OPERATOR_CLASSES["CHANGE_RATIO"]) and \
            ("DIFF" not in OPERATOR_CLASSES or operator_class != OPERATOR_CLASSES["DIFF"]):
        return -1
    
    
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy

    operands = get_operands(derivation)


    first_operand, second_operand = operands[0], operands[1]
    answer_from = answer_mapping.keys()
    table_answer_coordinates = None
    paragraph_answer_coordinates = None
    if "table" in answer_from:
        table_answer_coordinates = answer_mapping["table"]
    if "paragraph" in answer_from:
        paragraph_answer_coordinates = answer_mapping["paragraph"]

    table_answer_nums, paragraph_answer_nums = get_answer_nums(table_answer_coordinates, paragraph_answer_coordinates)
    if (table_answer_nums + paragraph_answer_nums) < 2:
        # print("the same number to skip it: derivation")
        raise RuntimeError(f" skip this the derivation is {derivation} ")
    if table_answer_nums == 2:
        answer_coordinates = answer_mapping["table"]
        answer_coordinates_copy = answer_coordinates.copy()
        answer_coordinates = [(answer_coordinate[0], answer_coordinate[1]) for answer_coordinate in
                              answer_coordinates_copy]
        del answer_coordinates_copy
        operand_one = to_number(table.iloc[answer_coordinates[0][0], answer_coordinates[0][1]])
        operand_two = to_number(table.iloc[answer_coordinates[1][0], answer_coordinates[1][1]])
        if str(operand_one) == str(first_operand):
            if answer_coordinates[0][0] < answer_coordinates[1][0]:
                return 0
            elif answer_coordinates[0][0] == answer_coordinates[1][0] and \
                    answer_coordinates[0][1] < answer_coordinates[1][1]:
                return 0
            else:
                return 1
        else:
            if answer_coordinates[0][0] > answer_coordinates[1][0]:
                return 1
            elif answer_coordinates[0][0] == answer_coordinates[1][0] and \
                    answer_coordinates[0][1] > answer_coordinates[1][1]:
                return 1
            else:
                return 0
    elif paragraph_answer_nums == 2:

        paragraph_mapping_orders = list(answer_mapping["paragraph"].keys())
        if len(paragraph_mapping_orders) == 1:
            answer_one_order, answer_two_order = (paragraph_mapping_orders[0], paragraph_mapping_orders[0])
            answer_one_start = answer_mapping["paragraph"][answer_one_order][0][0]
            answer_one_end = answer_mapping["paragraph"][answer_one_order][0][1]
            answer_two_start = answer_mapping["paragraph"][answer_two_order][1][0]
            answer_two_end = answer_mapping["paragraph"][answer_two_order][1][1]
        else:
            answer_one_order = paragraph_mapping_orders[0]
            answer_two_order = paragraph_mapping_orders[1]
            answer_one_start = answer_mapping["paragraph"][answer_one_order][0][0]
            answer_one_end = answer_mapping["paragraph"][answer_one_order][0][1]
            answer_two_start = answer_mapping["paragraph"][answer_two_order][0][0]
            answer_two_end = answer_mapping["paragraph"][answer_two_order][0][1]
        operand_one = to_number(paragraphs[int(answer_one_order)][answer_one_start:answer_one_end])
        operand_two = to_number(paragraphs[int(answer_two_order)][answer_two_start:answer_two_end])
        if operand_one == first_operand:
            if answer_one_order < answer_two_order:
                return 0
            elif answer_one_order == answer_two_order and answer_one_start < answer_two_start:
                return 0
            else:
                return 1
        else:
            if answer_one_order > answer_two_order:
                return 1
            elif answer_one_order == answer_two_order and answer_one_start > answer_two_start:
                return 1
            else:
                return 0
    else:
        answer_coordinates = answer_mapping["table"]
        operand_one = to_number(table.iloc[answer_coordinates[0][0], answer_coordinates[0][1]])
        paragraph_mapping_orders = list(answer_mapping["paragraph"].keys())
        answer_two_order = paragraph_mapping_orders[0]
        answer_two_start = answer_mapping["paragraph"][answer_two_order][0][0]
        answer_two_end = answer_mapping["paragraph"][answer_two_order][0][1]
        operand_two = to_number(paragraphs[int(answer_two_order)][answer_two_start:answer_two_end])
        if operand_one == first_operand:
            return 0
        else:
            return 1


def _concat(question_ids,
            question_split_tokens,
            
            table_ids,
            table_split_cell_tokens,
            table_tags,
            table_cell_index,
            table_cell_number_value,
            table_row_header_index, 
            table_col_header_index, 
            table_data_cell_index,
            same_row_data_cell_rel,
            same_col_data_cell_rel,
            data_cell_row_rel, 
            data_cell_col_rel,
            table_cell_number_value_with_time,
            row_tags, 
            col_tags,
            row_include_cells, 
            col_include_cells,
            all_cell_index,
            all_cell_token_index,
            
            paragraph_ids,
            paragraph_split_tokens,
            paragraph_tags,
            paragraph_index,
            paragraph_number_value,
            sep_start,
            sep_end,
            question_length_limitation,
            passage_length_limitation,
            max_pieces,):
    
    import pdb
#     pdb.set_trace()
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    tags = torch.zeros_like(input_ids)
    
    
    ## add question mask 
    question_mask = torch.zeros_like(input_ids)
    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_ids = question_ids[:question_length_limitation]
            question_split_tokens = question_split_tokens[:question_length_limitation]
    question_ids = [sep_start] + question_ids + [sep_end]
    question_split_tokens = ['<s>'] + question_split_tokens + ['</s>']
    question_length = len(question_ids)
    question_mask[0, 1:question_length-1] = 1
    
    
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            passage_split_tokens = table_split_cell_tokens[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
            max_cell_num = table_cell_index[passage_length_limitation-1]
            table_row_header_index, table_col_header_index, table_data_cell_index,same_row_data_cell_rel,\
    same_col_data_cell_rel, data_cell_row_rel,data_cell_col_rel,table_cell_number_value_with_time, row_tags, col_tags,row_include_cells, col_include_cells, all_cell_index, all_cell_token_index=clip_table(max_cell_num,table_row_header_index, 
                                                                table_col_header_index,table_data_cell_index,
                                                                             same_row_data_cell_rel, 
                                                                             same_col_data_cell_rel, 
                                                                             data_cell_row_rel, 
                                                                             data_cell_col_rel,
                                                                             table_cell_number_value_with_time,
                                                                             row_tags,
                                                                             col_tags,
                                                                             row_include_cells, 
                                                                             col_include_cells,
                                                                             all_cell_index, 
                                                                             all_cell_token_index
                                                                           )
            assert len(row_tags) == len(row_include_cells)
            assert len(col_tags) == len(col_include_cells)
            
        elif len(table_ids) + len(paragraph_ids) > passage_length_limitation:
            passage_ids = table_ids + [sep_end] + paragraph_ids
            passage_split_tokens = table_split_cell_tokens + ['</s>'] + paragraph_split_tokens
            passage_ids = passage_ids[:passage_length_limitation]
            passage_split_tokens = passage_split_tokens[:passage_length_limitation]
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length
            
        else:
            passage_ids = table_ids + [sep_end] + paragraph_ids
            passage_split_tokens = table_split_cell_tokens + ['</s>'] + paragraph_split_tokens
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids) + 1
    else:
        passage_ids = table_ids + [sep_end] + paragraph_ids
        passage_split_tokens = table_split_cell_tokens + ['</s>'] + paragraph_split_tokens
    passage_ids = passage_ids + [sep_end]
    passage_split_tokens = passage_split_tokens + ['</s>']
    
    
    ## input
    
    input_ids_list = question_ids + passage_ids
    
    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    attention_mask = input_ids != 0
    table_mask[0, question_length:question_length + table_length] = 1
    table_index[0, question_length:question_length + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))
    tags[0, question_length:question_length + table_length] = torch.from_numpy(np.array(table_tags[:table_length]))
    if paragraph_length > 1:
        paragraph_mask[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = 1
        paragraph_index[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(in_paragraph_index[:paragraph_length - 1]))
        tags[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(paragraph_tags[:paragraph_length - 1]))
        
    del in_table_cell_index
    del in_paragraph_index
    
    input_tokens = question_split_tokens + passage_split_tokens
    
#     pdb.set_trace()
    
    return input_ids, input_ids_list, input_tokens, question_mask, attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, \
           table_mask, table_cell_number_value, table_index, tags, input_segments, table_row_header_index,                  table_col_header_index, table_data_cell_index,same_row_data_cell_rel,\
           same_col_data_cell_rel, data_cell_row_rel,data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index


def clip_table(max_table_cell_num, table_row_header_index, table_col_header_index, table_data_cell_index,
                same_row_data_cell_rel, same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel,table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index):
    '''
    clip for overlength table cell
    '''
    table_row_header_index = list(filter(lambda x:x[0]<=max_table_cell_num, table_row_header_index))
    table_col_header_index = list(filter(lambda x:x[0]<=max_table_cell_num, table_col_header_index))
    table_data_cell_index = list(filter(lambda x:x[0]<=max_table_cell_num, table_data_cell_index))
    all_cell_index = list(filter(lambda x:x[0]<=max_table_cell_num, all_cell_index))
    
    
    same_row_data_cell_rel = clip_list(same_row_data_cell_rel, max_table_cell_num)
    same_col_data_cell_rel = clip_list(same_col_data_cell_rel, max_table_cell_num)
#     import pdb
#     pdb.set_trace()
    table_cell_number_value_with_time = table_cell_number_value_with_time[:max_table_cell_num]
    
    data_cell_row_rel = clip_dict(data_cell_row_rel, max_table_cell_num)
    data_cell_col_rel = clip_dict(data_cell_col_rel, max_table_cell_num)
    
    ## clip row and column tags
    row_include_cells = clip_dict(row_include_cells, max_table_cell_num)
    col_include_cells = clip_dict(col_include_cells, max_table_cell_num)
    
    row_tags = row_tags[:len(row_include_cells.keys())]
    col_tags = col_tags[:len(col_include_cells.keys())]
    
    all_cell_token_index = all_cell_token_index[:len(all_cell_index)]
    
    return table_row_header_index, table_col_header_index,table_data_cell_index,same_row_data_cell_rel, same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags,row_include_cells, col_include_cells, all_cell_index, all_cell_token_index

def clip_dict(clipped_dict, max_value):
    '''
    clip for overlength table cell
    '''
    new_clipped_dict = {}
    for key, value in clipped_dict.items():
        if key<=max_value and max(value)<=max_value:
            new_clipped_dict.update({key:value})
        elif key<=max_value:
#             import pdb
#             pdb.set_trace()
            clip_value = list(filter(lambda x:x<=max_value, value))
            if len(clip_value)>0:
                new_clipped_dict.update({key:clip_value})
        else:
            continue
    del clipped_dict
    return new_clipped_dict
    
def clip_list(clipped_list, max_value):
    new_clipped_list = []
    for item in clipped_list:
        item = list(filter(lambda x:x<=max_value, item))
        if len(item)>0:
            new_clipped_list.append(item)
    return new_clipped_list

    


def _test_concat(question_ids,
                table_ids,
                table_tags,
                table_cell_index,
                table_cell_number_value,
                 
                table_row_header_index,
                table_col_header_index,
                table_data_cell_index,
                same_row_data_cell_rel,
                same_col_data_cell_rel,
                data_cell_row_rel, 
                data_cell_col_rel,
                table_cell_number_value_with_time, 
                row_tags,
                col_tags, 
                row_include_cells, 
                col_include_cells, 
                all_cell_index,
                
                 
                paragraph_ids,
                paragraph_tags,
                paragraph_index,
                paragraph_number_value,
                sep_start,
                sep_end,
                question_length_limitation,
                passage_length_limitation,
                max_pieces,):
    in_table_cell_index = table_cell_index.copy()
    in_paragraph_index = paragraph_index.copy()
    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)
    paragraph_index = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    table_index = torch.zeros_like(input_ids)
    tags = torch.zeros_like(input_ids)
    
    ## add question mask
    question_mask = torch.zeros_like(input_ids)
    

    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_ids = question_ids[:question_length_limitation]
    question_ids = [sep_start] + question_ids + [sep_end]
    question_length = len(question_ids)
    
    ## add question mask
    question_mask[0, 1:question_length-1] = 1
    
    table_length = len(table_ids)
    paragraph_length = len(paragraph_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            passage_ids = table_ids[:passage_length_limitation]
            table_length = passage_length_limitation
            paragraph_length = 0
            max_cell_num = table_cell_index[passage_length_limitation-1]

            table_row_header_index, table_col_header_index, table_data_cell_index,same_row_data_cell_rel,\
    same_col_data_cell_rel, data_cell_row_rel,data_cell_col_rel,table_cell_number_value_with_time,row_tags,col_tags, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index =clip_table(max_cell_num,table_row_header_index, 
                                                                table_col_header_index,table_data_cell_index,
                                                                             same_row_data_cell_rel, 
                                                                             same_col_data_cell_rel, 
                                                                             data_cell_row_rel, 
                                                                             data_cell_col_rel,
                                                                             table_cell_number_value_with_time,
                                                                             row_tags,
                                                                             col_tags, 
                                                                             row_include_cells, 
                                                                             col_include_cells, 
                                                                             all_cell_index,
                                                                             all_cell_token_index
                                                                          )
#             import pdb
#             pdb.set_trace()
#             print("overlength")
            
        elif len(table_ids) + len(paragraph_ids) > passage_length_limitation:
            passage_ids = table_ids + [sep_end] + paragraph_ids
            passage_ids = passage_ids[:passage_length_limitation]
            table_length = len(table_ids)
            paragraph_length = passage_length_limitation - table_length
#             print("overlength")
        else:
            passage_ids = table_ids + [sep_end] + paragraph_ids
            table_length = len(table_ids)
            paragraph_length = len(paragraph_ids) + 1
    else:
        passage_ids = table_ids + [sep_end] + paragraph_ids

    passage_ids = passage_ids + [sep_end]

    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_ids[0, question_length:question_length + len(passage_ids)] = torch.from_numpy(np.array(passage_ids))
    attention_mask = input_ids != 0
    table_mask[0, question_length:question_length + table_length] = 1
    table_index[0, question_length:question_length + table_length] = \
        torch.from_numpy(np.array(in_table_cell_index[:table_length]))
    tags[0, question_length:question_length + table_length] = torch.from_numpy(np.array(table_tags[:table_length]))
    if paragraph_length > 1:
        paragraph_mask[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = 1
        paragraph_index[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(in_paragraph_index[:paragraph_length - 1]))
        tags[0, question_length + table_length + 1:question_length + table_length + paragraph_length] = \
            torch.from_numpy(np.array(paragraph_tags[:paragraph_length - 1]))
    del in_table_cell_index
    del in_paragraph_index
    return input_ids, question_mask, attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, \
           table_mask, table_cell_number_value, table_index, tags, input_segments, table_row_header_index,\
table_col_header_index, table_data_cell_index,same_row_data_cell_rel,same_col_data_cell_rel,\
data_cell_row_rel,data_cell_col_rel,table_cell_number_value_with_time, row_tags,  col_tags,  row_include_cells, col_include_cells, all_cell_index, all_cell_token_index
                                                                             
                                                                            
                                                                             
                                                                             

"""
instance format:
input_ids: np.array[1, 512]. The input ids.
attention_mask: np.array[1, 512]. The attention_mask to denote whether a id is real or padded.
token_type_ids: np.array[1, 512, 3]. 
    The special tokens needed by tapas within following orders: segment_ids, column_ids, row_ids.
tags_label: np.array[1, 512]. The tag ground truth indicate whether this id is part of the answer.
paragraph_mask: np.array[1, 512]. 1 for ids belongs to paragraph, 0 for others
paragraph_word_piece_mask: np.array[1, 512]. 0 for sub-token, 1 for non-sub-token or the start of sub-token
paragraph_number_value: np.array[1, 512]. nan for no-numerical words and number for numerical words extract from current word. i.e.: $460 -> 460
table_number_value: np.array[1, max_num_columns*max_num_rows]. Definition is the same as above.
paragraph_number_mask: np.array[1, 512]. 0 for non numerical token and 1 for numerical token.
table_number_mask: np.array[1, max_num_columns*max_num_rows]. 0 for non numerical token and 1 for numerical token.
paragraph_index: np.array[1, 512], used to apply token-lv reduce-mean after getting sequence_output
number_order_label: int. The operator calculating order.
operator_label:  int. The operator ground truth.
scale_label: int. The scale ground truth.
answer: str. The answer used to calculate metrics.
"""


class TagTaTQAReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>", op_mode:int=8,
                 ablation_mode:int=0):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep_start = self.tokenizer._convert_token_to_id(sep)
        self.sep_end = self.tokenizer._convert_token_to_id(sep)
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.op_mode=op_mode
        if ablation_mode == 0:
            self.OPERATOR_CLASSES=OPERATOR_CLASSES_
        elif ablation_mode == 1:
            self.OPERATOR_CLASSES=get_op_1(op_mode)
        elif ablation_mode == 2:
            self.OPERATOR_CLASSES=get_op_2(op_mode)
        else:
            self.OPERATOR_CLASSES=get_op_3(op_mode)
        
        self.logical_former = LogicalFormer(GRAMMER_CLASS, GRAMMER_ID)
        
        

    def _make_instance(self, input_ids, input_tokens, question_mask, attention_mask, token_type_ids, paragraph_mask, 
                       table_mask,paragraph_number_value, table_cell_number_value, paragraph_index, 
                       table_cell_index,number_order_label, tags_ground_truth, operator_ground_truth, 
                       scale_ground_truth,paragraph_tokens, table_cell_tokens, answer_dict, 
                       question_id,table_row_header_index,  
                       table_col_header_index,table_data_cell_index,same_row_data_cell_rel,
                       same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, row_tags, col_tags,row_include_cells, col_include_cells, all_cell_index, logical_forms):
        
        import pdb
#         pdb.set_trace()
        
        return {
            "input_ids": np.array(input_ids),
            "input_tokens":input_tokens,
            "question_mask":np.array(question_mask),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "number_order_label": int(number_order_label),
            "tag_labels": np.array(tags_ground_truth),
            "operator_label": int(operator_ground_truth),
            "scale_label": int(scale_ground_truth),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            
            ## add table structure
            "table_row_header_index":table_row_header_index,  
            "table_col_header_index":table_col_header_index, 
            "table_data_cell_index":table_data_cell_index, 
            "same_row_data_cell_rel":same_row_data_cell_rel, 
            "same_col_data_cell_rel":same_col_data_cell_rel, 
            "data_cell_row_rel":data_cell_row_rel, 
            "data_cell_col_rel":data_cell_col_rel,
            "row_tags":np.array(row_tags),
            "col_tags":np.array(col_tags),
            "row_include_cells":row_include_cells,
            "col_include_cells":col_include_cells,
            "all_cell_index":all_cell_index, 
            "logical_forms":logical_forms
        }

    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer:str, derivation: str, facts:list,  answer_mapping: Dict, scale: str, question_id:str):
        
        question_text = question.strip()
        operator_class = get_operator_class(derivation, answer_type, facts, answer,answer_mapping, scale, self.OPERATOR_CLASSES)
                                            
        scale_class = SCALE.index(scale)
        if operator_class is None:
            self.skip_count += 1
            return None

        table_cell_tokens, table_split_cell_tokens, table_ids, table_tags,table_cell_number_value,table_cell_index,\
        table_row_header_index, table_col_header_index, table_data_cell_index, \
        same_row_data_cell_rel,same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel,  table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index\
        =table_tokenize(table, self.tokenizer, answer_mapping, answer_type)
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"
        table = pd.DataFrame(table, dtype=np.str)
        column_relation = {}
        for column_name in table.columns.values.tolist():
            column_relation[column_name] = str(column_name)
        table.rename(columns=column_relation, inplace=True)
        
        paragraph_tokens, paragraph_split_tokens, paragraph_ids, paragraph_tags, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index= \
            paragraph_tokenize(question, paragraphs, self.tokenizer, answer_mapping, answer_type)

        question_split_tokens, question_ids = question_tokenizer(question_text, self.tokenizer)
       
        number_order_label = get_number_order_labels(paragraphs, table, derivation, operator_class,
                                                             answer_mapping, question_id, self.OPERATOR_CLASSES)



        example = {
                    "question_ids":question_ids, 
                    "question_split_tokens":question_split_tokens,

                    "table_ids":table_ids, 
                    "table_split_cell_tokens":table_split_cell_tokens,
                    "table_tags":table_tags, 
                    "table_cell_index":table_cell_index, 
                    "table_cell_number_value":table_cell_number_value,
                    "table_row_header_index":table_row_header_index, 
                    "table_col_header_index":table_col_header_index,
                    "table_data_cell_index":table_data_cell_index,
                    "same_row_data_cell_rel":same_row_data_cell_rel,
                    "same_col_data_cell_rel":same_col_data_cell_rel, 
                    "data_cell_row_rel":data_cell_row_rel, 
                    "data_cell_col_rel":data_cell_col_rel,
                    "table_cell_number_value_with_time":table_cell_number_value_with_time,
                    "row_tags":row_tags, 
                    "col_tags":col_tags, 
                    "row_include_cells":row_include_cells, 
                    "col_include_cells":col_include_cells, 
                    "all_cell_index":all_cell_index, 
                    "all_cell_token_index":all_cell_token_index,

                    "paragraph_ids":paragraph_ids, 
                    "paragraph_split_tokens":paragraph_split_tokens,
                    "paragraph_tags":paragraph_tags, 
                    "paragraph_index":paragraph_index, 
                    "paragraph_number_value":paragraph_number_value, 
                    "sep_start":self.sep_start, 
                    "sep_end":self.sep_end, 
                    "question_length_limitation":self.question_length_limit, 
                    "passage_length_limitation":self.passage_length_limit, 
                    "max_pieces":self.max_pieces
        }


        input_ids, input_tokens, question_mask, attention_mask, paragraph_mask, paragraph_number_value, paragraph_index, \
        table_mask, table_number_value, table_index, tags, token_type_ids,table_row_header_index, \
        table_col_header_index,table_data_cell_index,same_row_data_cell_rel,\
                    same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index, all_cell_token_index = \
            _concat(**example)

        # get_logical_form

        table_start = int(torch.sum(question_mask[0])) + 2
        instance = {
            "question":question,
            "tokenizer":self.tokenizer,
            "table":table,
            "input_tokens":input_tokens,
            "tags":tags,
            "table_start":table_start,
            "table_mask":table_mask,
            "paragraph_mask":paragraph_mask,
            "table_number_value":table_number_value,
            "all_cell_index":all_cell_index, 
            "all_cell_token_index":all_cell_token_index,

            "answer_from":answer_from,
            "answer_type":answer_type, 
            "answer":answer,
            "derivation":derivation, 
            "facts":facts,
            "answer_mapping":answer_mapping,
            "scale":scale,
        }
        
        logical_forms = []
#         if question == "What is the difference between 2019 average rate of inflation and 2019 average rate of increase in salaries?":
#             import pdb
#             pdb.set_trace()
        logical_forms = self.logical_former.get_logical_forms(**instance)
    
    
        
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from}
        return self._make_instance(input_ids, input_tokens, question_mask, attention_mask, token_type_ids, paragraph_mask, table_mask,paragraph_number_value, table_number_value, paragraph_index, table_index,number_order_label, tags, operator_class, scale_class,paragraph_tokens, table_cell_tokens, answer_dict, question_id,table_row_header_index, table_col_header_index,table_data_cell_index,same_row_data_cell_rel,
                    same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, row_tags, col_tags,row_include_cells, col_include_cells, all_cell_index, logical_forms)
    


    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances = []
        key_error_count = 0
        index_error_count = 0
        assert_error_count = 0
        lf_examples= []
        error_examples = []
        import pdb
#         pdb.set_trace()
        error_case = 0
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']
            
            for question_answer in questions:
                try:
                    question = question_answer["question"].strip()
                    answer_type = question_answer["answer_type"]
                    derivation = question_answer["derivation"]
                    answer = question_answer["answer"]
#                     import pdb
#                     pdb.set_trace()
                    answer_mapping = question_answer["mapping"]
                    facts = question_answer["facts"]
                    answer_from = question_answer["answer_from"]
                    scale = question_answer["scale"]
                    instance = self._to_instance(question, table, paragraphs, answer_from,
                                    answer_type, answer, derivation, facts, answer_mapping, scale, question_answer["uid"])
                    if instance is not None:
                        instances.append(instance)
                        
                        input_tokens = instance["input_tokens"]
                        logical_forms = instance["logical_forms"]
                        logical_forms_2 = []
                        import pdb
#                         pdb.set_trace()
                        for logical_form in logical_forms:
                            logical_form_2 = []
                            argments_stack = []
                            for item in logical_form:
                                if isinstance(item, str):
                                    logical_form_2.append(item)
                                
                                if isinstance(item, int):
                                    argments_stack.append(item)
                                
                                if len(argments_stack)==2:
                                    s = argments_stack[0]
                                    e = argments_stack[1]
                                    argments_stack = []
                                    logical_form_2.append(" ".join([token.strip(USTRIPPED_CHARACTERS) for token in input_tokens[s:e]]) + " ")
                            logical_forms_2.append(logical_form_2)
                        try:
                            answer_from_mapping = []
                            if 'table' in answer_mapping:
                                answer_table_cell = answer_mapping["table"]
                                answer_table_cell = [table[cell[0]][cell[1]] for cell in answer_table_cell]
                                answer_from_mapping.extend(answer_table_cell)
                            if 'paragraph' in answer_mapping:
                                import pdb
                               
                                answer_span = []
                                for key, value in answer_mapping['paragraph'].items():
                                    text = paragraphs[int(key)-1]['text']
                                    for item in value:
                                        answer_span.append(text[item[0]:item[1]])
                                answer_from_mapping.extend(answer_span)
            
                            lf_example = {
                                "question":question, 
                                "table":table, 
                                "paragraphs":paragraphs, 
                                "answer_from":answer_from, 
                                "answer_type":answer_type, 
                                "answer":answer, 
                                "derivation":derivation, 
                                "facts":facts, 
                                "answer_mapping":answer_mapping, 
                                "answer_from_mapping":answer_from_mapping,
                                "logical_forms": [" ".join([str(item) for item in logical_form]) for logical_form in logical_forms],
                                "logical_forms_2": [" ".join([str(item) for item in logical_form]) for logical_form in logical_forms_2]
                            }
                        except:
                            import pdb
                            pdb.set_trace()
                            
                        if len(logical_forms)==0:
                            error_case+=1
                            error_examples.append(lf_example)
                            continue
                        
                        lf_examples.append(lf_example)
                       
                except RuntimeError as e :
                    print(f"run time error:{e}" )
                    # print(question_answer["uid"])
                except KeyError:
                    key_error_count += 1
                    # print(question_answer["uid"])
                    print("KeyError. Total Error Count: {}".format(key_error_count))
                except IndexError:
                    index_error_count += 1
                    # print(question_answer["uid"])
                    print("IndexError. Total Error Count: {}".format(index_error_count))
                except AssertionError:
                    assert_error_count += 1
                    # print(question_answer["uid"])
                    print("AssertError. Total Error Count: {}".format(assert_error_count))
#         pdb.set_trace()
        with open('./logical_forms_train.json', 'w') as f:
            json.dump(lf_examples, f, indent=4)
        f.close()
        print("lf_examples",len(lf_examples))
        with open('./error_logical_forms_train.json', 'w') as f:
            json.dump(error_examples, f, indent=4)
        f.close()
        
        print("finish the logocal forms")
        print("error_case:", error_case)
        return instances

class TagTaTQATestReader(object):
    def __init__(self, tokenizer,
                 passage_length_limit: int = None, question_length_limit: int = None, sep="<s>",
                 ablation_mode=0, op_mode=0):
        self.max_pieces = 512
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep_start = self.tokenizer._convert_token_to_id(sep)
        self.sep_end = self.tokenizer._convert_token_to_id(sep)
        tokens = self.tokenizer._tokenize("Feb 2 Nov")
        self.skip_count = 0
        self.ablation_mode = ablation_mode
        if self.ablation_mode == 3:
            self.OPERATOR_CLASSES=get_op_3(op_mode)
        self.op_count = {"Span-in-text":0, "Cell-in-table":0, "Spans":0, "Sum":0, "Count":0, "Average":0,
                         "Multiplication":0, "Division":0, "Difference":0, "Change ratio":0}
        self.scale_count = {"":0, "thousand":0, "million":0, "billion":0, "percent":0}

    def _make_instance(self, input_ids, question_mask, attention_mask, token_type_ids, paragraph_mask, table_mask,
                        paragraph_number_value, table_cell_number_value, paragraph_index, table_cell_index,
                        tags_ground_truth, paragraph_tokens, table_cell_tokens, answer_dict, question_id,
                      table_row_header_index, table_col_header_index,table_data_cell_index,same_row_data_cell_rel,
                       same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells,col_include_cells,all_cell_index):
#         import pdb
#         pdb.set_trace()
        
        return {
            "input_ids": np.array(input_ids),
            "question_mask":np.array(question_mask),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "paragraph_mask": np.array(paragraph_mask),
            "table_mask": np.array(table_mask),
            "paragraph_number_value": np.array(paragraph_number_value),
            "table_cell_number_value": np.array(table_cell_number_value),
            "paragraph_index": np.array(paragraph_index),
            "table_cell_index": np.array(table_cell_index),
            "tag_labels": np.array(tags_ground_truth),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "answer_dict": answer_dict,
            "question_id": question_id,
            ## add table structure
            "table_row_header_index":table_row_header_index,  
            "table_col_header_index":table_col_header_index, 
            "table_data_cell_index":table_data_cell_index, 
            "same_row_data_cell_rel":same_row_data_cell_rel, 
            "same_col_data_cell_rel":same_col_data_cell_rel, 
            "data_cell_row_rel":data_cell_row_rel, 
            "data_cell_col_rel":data_cell_col_rel,
            "table_cell_number_value_with_time":np.array(table_cell_number_value_with_time),
            "row_tags":np.array(row_tags),
            "col_tags":np.array(col_tags),
            "row_include_cells":row_include_cells,
            "col_include_cells":col_include_cells,
            "all_cell_index":all_cell_index
        }

    def summerize_op(self, derivateion, answer_type, facts, answer, answer_mapping, scale):
        if answer_type == "span":
            if "table" in answer_mapping.keys():
                self.op_count["Cell-in-table"] += 1
                return "Cell-in-table"
            elif "paragraph" in answer_mapping.keys():
                self.op_count["Span-in-text"] += 1
                return "Span-in-text"
        elif answer_type == "multi-span":
            self.op_count["Spans"] += 1
            return "Spans"
        elif answer_type == "count":
            self.op_count["Count"] += 1
            return "Count"
        elif answer_type == "arithmetic":
            num_facts = facts_to_nums(facts)
            if not is_number(str(answer)):
                return ""
            if _is_change_ratio(num_facts, answer):
                self.op_count["Change ratio"] += 1
                return "Change ratio"
            elif _is_average(num_facts, answer):
                self.op_count["Average"] += 1
                return "Average"
            elif _is_sum(num_facts, answer):
                self.op_count["Sum"] += 1
                return "Sum"
            elif _is_times(num_facts, answer):
                self.op_count["Multiplication"] += 1
                return "Multiplication"
            elif _is_diff(num_facts, answer):
                self.op_count["Difference"] += 1
                return "Difference"
            elif _is_division(num_facts, answer):
                self.op_count["Division"] += 1
                return "Division"

    def _to_test_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer:str, answer_mapping, scale: str, question_id:str, derivation, facts):
        question_text = question.strip()
        if self.ablation_mode == 3:
            operator_class = get_operator_class(derivation, answer_type, facts, answer,
                                                answer_mapping, scale, self.OPERATOR_CLASSES)
            if not operator_class:
                self.skip_count += 1
                return None
        gold_op = self.summerize_op(derivation, answer_type, facts, answer, answer_mapping, scale)
        if gold_op is None:
            gold_op = "ignore"
        
        
        
        table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index,\
        table_row_header_index, table_col_header_index, table_data_cell_index,\
        same_row_data_cell_rel,same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time,row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index =\
        table_test_tokenize(question_id, table, self.tokenizer, answer_mapping, answer_type)

        
        paragraph_tokens, paragraph_ids, paragraph_tags, paragraph_word_piece_mask, paragraph_number_mask, \
                paragraph_number_value, paragraph_index, paragraph_mapping_content = \
            paragraph_test_tokenize(question, paragraphs, self.tokenizer, answer_mapping, answer_type)
        _, question_ids = question_tokenizer(question_text, self.tokenizer)

        input_ids, question_mask, attention_mask, paragraph_mask, paragraph_number_value, paragraph_index,\
        table_mask, table_number_value, table_index, tags, token_type_ids, table_row_header_index,\
        table_col_header_index,table_data_cell_index,same_row_data_cell_rel,\
        same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index = \
            _test_concat(question_ids, table_ids, table_tags, table_cell_index, table_cell_number_value,
             table_row_header_index,table_col_header_index,table_data_cell_index,same_row_data_cell_rel,
                    same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel, table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index,
                    paragraph_ids, paragraph_tags, paragraph_index, paragraph_number_value,
                    self.sep_start, self.sep_end, self.question_length_limit,
                    self.passage_length_limit, self.max_pieces)
        


        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from,
                       "gold_op": gold_op, "gold_scale":scale}
        self.scale_count[scale] += 1

        
        
        return self._make_instance(input_ids, question_mask, attention_mask, token_type_ids, paragraph_mask, table_mask,paragraph_number_value, table_number_value, paragraph_index, table_index, tags, paragraph_tokens,
            table_cell_tokens, answer_dict, question_id, table_row_header_index,
                                   table_col_header_index,table_data_cell_index,same_row_data_cell_rel, 
                                   same_col_data_cell_rel, data_cell_row_rel, data_cell_col_rel,table_cell_number_value_with_time, row_tags, col_tags, row_include_cells, col_include_cells, all_cell_index,)


    def _read(self, file_path: str):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        print("Reading the tatqa dataset")
        instances = []
        index_error_count = 0
        assert_error_count = 0
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']

            for question_answer in questions:
                try:
                    question = question_answer["question"].strip()
                    answer_type = question_answer["answer_type"]
                    answer = question_answer["answer"]
                    answer_from = question_answer["answer_from"]
                    answer_mapping = question_answer["mapping"]
                    scale = question_answer["scale"]
                    derivation = question_answer['derivation']
                    facts = question_answer['facts']
                    instance = self._to_test_instance(question, table, paragraphs, answer_from,
                                    answer_type, answer, answer_mapping, scale, question_answer["uid"], derivation, facts)
                    if instance is not None:
                        instances.append(instance)
                except RuntimeError:
                    print(question_answer["uid"])
                except IndexError:
                    index_error_count += 1
                    print(question_answer["uid"])
                    print("IndexError. Total Error Count: {}".format(index_error_count))
                except AssertionError:
                    assert_error_count += 1
                    print(question_answer["uid"])
                    print("AssertError. Total Error Count: {}".format(assert_error_count))
                except KeyError:
                    continue
        print(self.op_count)
        print(self.scale_count)
        self.op_count = {"Span-in-text": 0, "Cell-in-table": 0, "Spans": 0, "Sum": 0, "Count": 0, "Average": 0,
                         "Multiplication": 0, "Division": 0, "Difference": 0, "Change ratio": 0}
        self.scale_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        return instances

# ### Beginning of everything related to segmented tensors ###

class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object

class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
            .type(torch.float)
            .floor()
            .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def reduce_mean(values, index, name="segmented_reduce_mean"):
    """
    Averages a tensor over its segments.
    Outputs 0 for empty segments.
    This operations computes the mean over segments, with support for:
        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.
    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.
    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used
    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    return _segment_reduce(values, index, "mean", name)

def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)

def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])

def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()) :]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index



# def years_detector(text):
#     a = text.split(',')
#     import re
#     res = []
#     for item in a:
#         res += [i for i in re.findall("(20\d{2})",item) if int(i) >= 2000 and int(i)<=2021 ]
#     return res

def years_detector(text):
    import re
    res = [i for i in re.findall("(20\d{2})",text) if int(i) >= 2000 and int(i)<=2022 ]
    return res

    
if __name__ == "__main__":
#     x= "sd asf2000 as 2012 asd2222 123540we0s"
#     x = "$15,386"
    x = "April 26, 2019"
    res = years_detector(x)
    print(res)

    
 