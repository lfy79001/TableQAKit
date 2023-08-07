from tatqa_utils import to_number, is_number
# from data_util import *
from itertools import product
# from tag_op.data.data_util import facts_to_nums, _is_average, _is_change_ratio, _is_sum, _is_times, _is_diff,_is_division
import numpy as np
import re

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

# GRAMMER_CLASS = {
#     "CELL(":0,
#     "CELL_VALUE(":1,
#     "SPAN(":2,
#     "VALUE(":3,
#     "COUNT(":4,
#     "ARGMAX(":5,
#     "ARGMIN(":6,
#     "ROW_KEY_VALUE(":7,
#     "COL_KEY_VALUE(":8,
#     "SUM(":9,
#     "DIFF(":10,
#     "DIV(":11,
#     "AVG(":12,
#     "CHANGE_R(":13,
#     "MULTI-SPAN(":14,
#     "TIMES(":15,
#     ")":16,
#     "[":17,
#     "]":18,
# } 


GRAMMER_CLASS = {
    "CELL(":0,
    "CELL_VALUE(":1,
    "SPAN(":2,
    "VALUE(":3,
    "COUNT(":4,
    "ARGMAX(":5,
    "ARGMIN(":6,
    "KEY_VALUE(":7,
#     "ROW_KEY_VALUE(":7,
#     "COL_KEY_VALUE(":8,
    "SUM(":8,
    "DIFF(":9,
    "DIV(":10,
    "AVG(":11,
    "CHANGE_R(":12,
    "MULTI-SPAN(":13,
    "TIMES(":14,
    ")":15,
#     "[":17,
#     "]":18,
} 

AUX_NUM = {
    "0":16,
    "1":17
}


GRAMMER_ID = dict(zip(GRAMMER_CLASS.values(), GRAMMER_CLASS.keys()))
AUX_NUM_ID =  dict(zip(AUX_NUM.values(), AUX_NUM.keys()))
OP_ID = {**GRAMMER_ID, **AUX_NUM_ID}


SCALE_CLASS={
#     SCALE = ["", "thousand", "million", "billion", "percent"]
    "THOUNSAND(":18,
    "MILLION(":19,
    "BILLION(":20,
    "PERCENT(":21,
    "NONE(":22,
}

SCALE = ["", "thousand", "million", "billion", "percent"]
SCALECLASS = ["NONE(", "THOUNSAND(", "MILLION(", "BILLION(", "PERCENT("]
SCALE2CLASS = dict(zip(SCALE, SCALECLASS))


# GRAMMER_ID = dict(zip(GRAMMER_CLASS.values(), GRAMMER_CLASS.keys()))
USTRIPPED_CHARACTERS = ''.join([u"Ġ"])


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def string_tokenizer(string: str, tokenizer):
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


class LogicalFormer(object):
    def __init__(self, GRAMMER_CLASS, GRAMMER_ID):
        self.GRAMMER_CLASS = GRAMMER_CLASS
        self.GRAMMER_ID = GRAMMER_ID
        
    def get_logical_forms(self, question, tokenizer, table, input_tokens, tags, table_start, table_mask, paragraph_mask, table_number_value, all_cell_index, all_cell_token_index, answer_from, answer_type, answer, derivation, facts, answer_mapping, scale):
#         pass
        logical_forms = []
        try:
            if answer_type == "span":
                if derivation=="":
                     logical_forms.extend(self.get_single_span_logical_forms(tokenizer, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, answer, input_tokens))
#                     pass
                else:
                    logical_form_tmp = self.get_span_comparison_logical_forms(table_start, table_number_value, derivation, all_cell_index, all_cell_token_index, answer_mapping, answer, input_tokens)
                    if len(logical_form_tmp)>0:
                        logical_forms.extend(logical_form_tmp)
                    else:
                        ## Bottoming
                        logical_forms.extend(self.get_single_span_logical_forms(tokenizer, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, answer, input_tokens))
#                     pass
                
            elif answer_type == "multi-span":
                logical_forms.extend(self.get_multi_span_logical_forms(tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, table_mask, paragraph_mask,derivation, answer_mapping, answer, input_tokens))
#                 #pass
            elif answer_type == "count":
                logical_forms.extend(self.get_count_logical_forms(table_start, all_cell_index, all_cell_token_index, tags,table_mask, paragraph_mask, answer_mapping, answer, derivation, input_tokens))
                #pass
            elif answer_type == "arithmetic":
                logical_forms_tmp = self.get_arithemtic_logical_forms(question, tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, derivation, answer_type, facts, answer, answer_mapping, scale, input_tokens)
                if len(logical_forms_tmp)>0:
                    logical_forms.extend(logical_forms_tmp)
#                 #pass
            
        except KeyError:
            logical_forms = []
            
#         for id, logical_form in enumerate(logical_forms):
#             logical_form = [SCALE2CLASS[scale]] + logical_form + [")"]
#             logical_forms[id] = logical_form
        
        return logical_forms
    
            
            
    def find_cell(self, all_cell_index, pos):
        ## return the index  given the row/col
        for idx, item in enumerate(all_cell_index):
            if item[1] == pos[0]  and item[2] == pos[1]:
                return idx
        return -1
    
    
    def get_single_span_logical_forms(self, tokenizer, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, answer, input_tokens):
        '''
        Deal for single span 
        '''
        logical_forms = []
        try:
            if "table" in answer_mapping.keys() and len(answer_mapping['table'])>0:
                answer_pos = (answer_mapping['table'][0][0], answer_mapping['table'][0][1])
                answer_cell_index = self.find_cell(all_cell_index, answer_pos)
                if answer_cell_index == -1:
                    import pdb
                    pdb.set_trace()
                answer_token_start, answer_token_end = all_cell_token_index[answer_cell_index][0]+table_start, all_cell_token_index[answer_cell_index][1] + table_start
#                 answer_token_start, answer_token_end = self.check_boundary_for_table(tokenizer, input_tokens, answer[0], answer_token_start, answer_token_end)
                
                logical_form = ["CELL("]
                logical_form.extend([answer_token_start, answer_token_end-1, ")"])
                logical_forms.append(logical_form)

            if "paragraph" in answer_mapping.keys() and len(answer_mapping['paragraph'])>0:
                
                tags_tmp = tags * paragraph_mask
                paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp)
                if len(paragraph_token_index_answer)>0:
                    logical_form = ["SPAN("]
                    answer_token_start , answer_token_end = paragraph_token_index_answer[0][0], paragraph_token_index_answer[0][1]-1
#                     answer_token_start, answer_token_end = self.check_boundary(tokenizer, input_tokens, answer[0], answer_token_start, answer_token_end)
                    
                    logical_form.extend([answer_token_start , answer_token_end, ")"])
                    logical_forms.append(logical_form)
        except:
            import pdb
            pdb.set_trace()
        return logical_forms
    
    
    
    def check_boundary_for_table(self, tokenizer, input_tokens, answer_str, answer_token_start, answer_token_end):
        import pdb
        checked_answer_tokens = [token.strip(USTRIPPED_CHARACTERS).lower() for token in input_tokens[answer_token_start:answer_token_end]]
        answer_tokens = [token.strip(USTRIPPED_CHARACTERS).lower() for token in answer_str.split()]
        
        if ''.join(checked_answer_tokens) == ''.join(answer_tokens):
            return answer_token_start, answer_token_end
        else:
            start = 0 
            
            for id, token in enumerate(checked_answer_tokens):
                if token in answer_str.lower():
                    start = id
                    break
            
            end = start
            
            while end+1 < len(checked_answer_tokens):
                if checked_answer_tokens[end+1] in answer_str.lower():
                    end+=1
                else:
                    break
            if ''.join(checked_answer_tokens[start:end+1])==''.join(answer_tokens):
                print("re correct")
                import pdb
#                 pdb.set_trace()
                return answer_token_start+start, answer_token_start+end+1
                
            else:
                return answer_token_start, answer_token_end
                
    def get_span_comparison_logical_forms(self, table_start, table_number_value, derivation, all_cell_index, all_cell_token_index, answer_mapping, answer, input_tokens):
        logical_forms = []
        try:
            if ">" in derivation:
                items = derivation.split(">")
                items = [to_number(item.replace("%", "")) for item in items]
                answer_str = answer[0]
                try:
                    items_index = [table_number_value.index(item) for item in items]
                    items_pos = [(all_cell_index[i][1], all_cell_index[i][2]) for i in items_index]
                    answer_pos = (answer_mapping['table'][0][0], answer_mapping['table'][0][1])
                except:
                    return logical_forms
                if answer_pos[0] in [item[0] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((item[0], answer_pos[1]))
                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMAX(", "["]
                    logical_form = ["ARGMAX("]


                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["COL_KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
                        logical_form.extend(["KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    
                    logical_forms.append(logical_form)

                if answer_pos[1] in [item[1] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((answer_pos[0], item[1]))

                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMAX(", "["]
                    logical_form = ["ARGMAX("]
                    
                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["ROW_KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
                        logical_form.extend(["KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    
                    logical_forms.append(logical_form)

            ## deal for argmax  
            elif "<" in derivation:
                import pdb
    #             #pdb.set_trace()
                items = derivation.split("<")
                items = [to_number(item.replace("%", "")) for item in items]
                answer_str = answer[0]
                try:
                    items_index = [table_number_value.index(item) for item in items]
                    items_pos = [(all_cell_index[i][1], all_cell_index[i][2]) for i in items_index]
                    answer_pos = (answer_mapping['table'][0][0], answer_mapping['table'][0][1])
                except:
                    return logical_forms
                
                if answer_pos[0] in [item[0] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((item[0], answer_pos[1]))
                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMIN(", "["]
                    logical_form = ["ARGMIN("]
                    
                    
                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["COL_KEY_VALUE(",  "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")",")"])
                        logical_form.extend(["KEY_VALUE(",  "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")",")"])
                        
                        
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    logical_forms.append(logical_form)

                if answer_pos[1] in [item[1] for item in items_pos]:
                    keys_pos = []
                    for item in items_pos:
                        keys_pos.append((answer_pos[0], item[1]))

                    key_cells_index = [self.find_cell(all_cell_index, key_pos) for key_pos in keys_pos]    
                    key_cells_token_index = [all_cell_token_index[idx] for idx in key_cells_index]
                    value_cells_index = [self.find_cell(all_cell_index, item_pos) for item_pos in items_pos]
                    value_cells_token_index = [all_cell_token_index[idx] for idx in value_cells_index]
                    key_value_cells_token_index_pair = zip(key_cells_token_index, value_cells_token_index)
#                     logical_form = ["ARGMIN(", "["]
                    logical_form = ["ARGMIN("]
                    
                    for item in key_value_cells_token_index_pair:
#                         logical_form.extend(["ROW_KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
                        logical_form.extend(["KEY_VALUE(", "CELL(", table_start+item[0][0], table_start+item[0][1]-1, ")", "CELL_VALUE(", table_start+item[1][0], table_start+item[1][1]-1, ")", ")"])
#                     logical_form.extend(["]", ')'])
                    logical_form.extend([')'])
                    logical_forms.append(logical_form)
            else:
                return logical_forms
        except:
            import pdb
            pdb.set_trace()
        return logical_forms
    
    def get_all_answer_pos_in_table(self, answer, answer_mapping, table):
        '''
        mapping is not repeated, but answer may include same items
        also, there may be some Multiple combinations in the answer mapping
        '''
        answer_cells_pos = []
        for item in answer:
            answer_cells_pos.append([])
            for answer_cell in answer_mapping["table"]:
                cell_str = table.iloc[answer_cell[0],answer_cell[1]]
                if item in cell_str:
                    answer_cells_pos[-1].append((answer_cell[0], answer_cell[1]))
        
        result = list(product(*answer_cells_pos))
        import pdb
#         pdb.set_trace()
        return result
    
    def get_all_answer_pos_in_text(self,tokenizer, answer, paragraph_token_evidence_index, input_tokens):
        
        answer_token_pos = []
        for answer_item in answer:
            answer_token_pos.append([])
            answer_item_split_token, _ = string_tokenizer(answer_item, tokenizer)
            normlized_answer_item_split_token = [token.strip(USTRIPPED_CHARACTERS) for token in answer_item_split_token]
            for paragraph_token_evidence_index_item in paragraph_token_evidence_index:
                item_tokens = input_tokens[paragraph_token_evidence_index_item[0]:paragraph_token_evidence_index_item[1]]
                normlized_item_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in item_tokens]
                if normlized_answer_item_split_token == normlized_item_tokens:
                    answer_token_pos[-1].append((paragraph_token_evidence_index_item[0], paragraph_token_evidence_index_item[1]))
        result = list(product(*answer_token_pos))
        import pdb
#         pdb.set_trace()
        return result
        
    def get_all_answer_pos_in_text_patch(self, tokenizer, answer, paragraph_token_index_answer, input_tokens):
        
        
        def find_index(index, len_sum):
            id = -1
            for id in range(len(len_sum)):
                if index == len_sum[id]:
                    return id
            return id
        
        pos = []
        
        checked_answer = [input_tokens[offset[0]:offset[1]] for offset in paragraph_token_index_answer]
        normalized_checked_answer = [ [token.strip(USTRIPPED_CHARACTERS).lower() for token in item]   for item in checked_answer]
        
        start = 0
        import pdb
        
        for answer_item in answer:
            tokenized_answer_item, _ = string_tokenizer(answer_item, tokenizer)
            normalized_tokenized_answer_item = [token.strip(USTRIPPED_CHARACTERS).lower() for token in tokenized_answer_item]
            index=-1
            for id, normalized_checked_answer_item in enumerate(normalized_checked_answer):
                index  = ''.join(normalized_checked_answer_item).find(''.join(normalized_tokenized_answer_item))
                if index != -1:
                    start = paragraph_token_index_answer[id][0]
                    break
                
            
            if index!=-1:
               
                start_index = index
                end_index = index + len(''.join(normalized_tokenized_answer_item))
                len_sum = [ len(''.join(normalized_checked_answer_item[:i])) for i in range(0, len(normalized_checked_answer_item)+1)]
                
                start_index = find_index(start_index, len_sum)

                end_index = find_index(end_index, len_sum)
                
                if start_index == -1 or end_index==-1:
                    return []
                else:
                    pos.append((start+start_index, start+end_index))
#         pdb.set_trace()
        
        if len(pos) >0:
            return [pos]
        else:
            return []
            
        
        
    def get_multi_span_logical_forms(self, tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, table_mask, paragraph_mask, derivation, answer_mapping, answer, input_tokens):
        logical_forms = []
        if "table" in answer_mapping.keys() and len(answer_mapping["table"])>0 and ("paragraph" not in answer_mapping.keys() or len(answer_mapping["paragraph"])==0):
            answer_cells_pos = answer_mapping["table"]
            all_possible_answer_cells_pos = []
            all_possible_answer_cells_pos.append(answer_cells_pos)
            if len(answer_cells_pos)!=len(answer):
                all_possible_answer_cells_pos= self.get_all_answer_pos_in_table(answer, answer_mapping, table) ## makeup 
#             if len(all_possible_answer_cells_pos) ==0:
#                 import pdb
#                 pdb.set_trace()
            for answer_cells_pos in all_possible_answer_cells_pos:
                answer_cells_index = [self.find_cell(all_cell_index, answer_cell_pos) for answer_cell_pos in answer_cells_pos]
                answer_cells_token_index = [(all_cell_token_index[idx][0], all_cell_token_index[idx][1]) for idx in answer_cells_index]
#                 logical_form = ["MULTI-SPAN(", "["]
                logical_form = ["MULTI-SPAN("]
                
                for item in answer_cells_token_index:
                    logical_form.extend(["CELL(", item[0]+table_start, item[1]+table_start-1, ")"])
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                
                logical_forms.append(logical_form)
            
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" not in answer_mapping.keys() or len(answer_mapping["table"])==0):
            
            import pdb
#             pdb.set_trace()
            
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
            all_possible_paragraph_token_index_answer = []
#             all_possible_paragraph_token_index_answer.append(paragraph_token_index_answer)
            
            if len(paragraph_token_index_answer)>=len(answer):
                all_possible_paragraph_token_index_answer = self.get_all_answer_pos_in_text(tokenizer, answer,paragraph_token_index_answer, input_tokens)
            
            if len(paragraph_token_index_answer) < len(answer):
                all_possible_paragraph_token_index_answer = self.get_all_answer_pos_in_text_patch(tokenizer, answer,paragraph_token_index_answer, input_tokens)
            
            if len(all_possible_paragraph_token_index_answer) ==0:
                all_possible_paragraph_token_index_answer = [paragraph_token_index_answer]
#                 import pdb
#                 pdb.set_trace()
            for paragraph_token_index_answer in all_possible_paragraph_token_index_answer:
#                 logical_form = ["MULTI-SPAN(", "["]
                logical_form = ["MULTI-SPAN("]
                for item in paragraph_token_index_answer:
                    logical_form.extend(["SPAN(", item[0], item[1]-1, ")"])
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                logical_forms.append(logical_form)

        
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" in answer_mapping.keys() and len(answer_mapping["table"])!=0):
            answer_cells_pos = answer_mapping["table"]
            answer_cells_index = [self.find_cell(all_cell_index, answer_cell_pos) for answer_cell_pos in answer_cells_pos]
            answer_cells_token_index = [(all_cell_token_index[idx][0], all_cell_token_index[idx][1]) for idx in answer_cells_index]
            
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
            
            if len(answer_cells_pos) + len(paragraph_token_index_answer) == len(answer):
#                 logical_form = ["MULTI-SPAN(", "["]
                logical_form = ["MULTI-SPAN("]
                for item in answer_cells_token_index:
                    logical_form.extend(["CELL(", item[0]+table_start, item[1]+table_start-1, ")"])
                for item in paragraph_token_index_answer:
                    logical_form.extend(["SPAN(", item[0], item[1]-1, ")"])
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                logical_forms.append(logical_form)
                
            else:
#                 import pdb
#                 pdb.set_trace()
                return logical_forms
        import pdb
        #pdb.set_trace()
        return logical_forms
                
    
    def get_count_logical_forms(self, table_start, all_cell_index, all_cell_token_index, tags,table_mask, paragraph_mask, answer_mapping, answer, derivation, input_tokens):
        logical_forms = []
        if "table" in answer_mapping.keys() and len(answer_mapping["table"])>0 and ("paragraph" not in answer_mapping.keys() or len(answer_mapping["paragraph"])==0):
            answer_cells_pos = answer_mapping["table"]
            answer_cells_index = [self.find_cell(all_cell_index, answer_cell_pos) for answer_cell_pos in answer_cells_pos]
            answer_cells_token_index = [(all_cell_token_index[idx][0], all_cell_token_index[idx][1]) for idx in answer_cells_index]
            if len(answer_cells_token_index) != int(answer):
                import pdb
                pdb.set_trace()
#             logical_form = ["COUNT(", "["]
            logical_form = ["COUNT("]
            
            for item in answer_cells_token_index:
                logical_form.extend(["CELL(", item[0]+table_start, item[1]+table_start-1, ")"])
#             logical_form.extend(["]", ")"])
            logical_form.extend([")"])
            logical_forms.append(logical_form)
        
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" not in answer_mapping.keys() or len(answer_mapping["table"])==0):
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
            if len(paragraph_token_index_answer)>0:
#                 if len(paragraph_token_index_answer) != int(answer):
#                     import pdb
#                     pdb.set_trace()
#                 logical_form = ["COUNT(", "["]
                logical_form = ["COUNT("]
                for item in paragraph_token_index_answer:
                    logical_form.extend(["SPAN(", item[0], item[1]-1, ")"])
                
#                 logical_form.extend(["]", ")"])
                logical_form.extend([")"])
                logical_forms.append(logical_form)
        
        if "paragraph" in answer_mapping.keys() and len(answer_mapping["paragraph"])>0 and ("table" in answer_mapping.keys() and len(answer_mapping["table"])!=0):
            return logical_forms
        
        return logical_forms

    
    def get_arithemtic_logical_forms(self, question, tokenizer, table, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, derivation, answer_type, facts, answer, answer_mapping, scale, input_tokens):
        grammer_space = ["[", "]", "(", ")", "+", "-", "*", "/"]
        ## mapping the derivation into the grammer_number_sequence
        logical_forms = []
        
#         if question =="What is the percentage increase / (decrease) in the loss from operations from 2018 to 2019?":
#             import pdb
#             pdb.set_trace()
        try:
#             if '[' in derivation:
#                 import pdb
#                 pdb.set_trace()
            grammer_number_sequence = get_operators_from_derivation(derivation, grammer_space)
            
            normalized_grammer_number_sequence = normalize_grammer_number_sequence(grammer_number_sequence, grammer_space)
            post_grammer_number_sequence = convert_inorder_to_postorder(normalized_grammer_number_sequence, grammer_space)
            lf = get_lf_from_post_grammer_number_sequence(post_grammer_number_sequence, grammer_space)
            
            '''
            transfer arguments to index
            '''
            lf = self.mapping_arithmatic_arguments_to_index(question, tokenizer, table, lf, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, GRAMMER_CLASS, input_tokens)
            logical_forms.extend(lf)
        except:
            import pdb
#             pdb.set_trace()
            return []
        
        return logical_forms
    
    
    def mapping_arithmatic_arguments_to_index(self, question, tokenizer, table, lf, table_start, all_cell_index, all_cell_token_index, tags, paragraph_mask, answer_mapping, GRAMMER_CLASS, input_tokens):
        '''
        mapping arguments into index
        '''
        arguments = []
        for idx, op in enumerate(lf):
            if op not in GRAMMER_CLASS and op!="0" and op!="1":
                arguments.append((idx, op))

        import pdb
        
        answer_table_cell_token_index = []
        paragraph_token_index_answer = []
        
        if "table" in answer_mapping.keys():
            answer_table_cell = answer_mapping["table"]
            answer_table_cell_index = [self.find_cell(all_cell_index, cell) for cell in answer_table_cell]
            answer_table_cell_token_index = [(table_start+all_cell_token_index[cell_index][0], table_start+all_cell_token_index[cell_index][1])  for cell_index in answer_table_cell_index]
        if "paragraph" in answer_mapping.keys():
            tags_tmp = tags * paragraph_mask
            paragraph_token_index_answer = self.find_evidence_from_tags(tags_tmp) ## not deal with multi-mentions
        
        all_op_index = []
#         pdb.set_trace()
        
#         if question == "What is the increase / (decrease) in the Fabless design companies from 2018 to 2019?":
#             import pdb
#             pdb.set_trace()
            
        try:
            for idx, op in arguments:
                op_index = []
                if '%' in op:
                    op = op.replace("%","")
                    op = op.strip()
                if len(answer_table_cell_token_index) > 0:
                    for j, item in enumerate(answer_table_cell):
#                         tokenized_item, _ =  string_tokenizer(table.iloc[item[0], item[1]], tokenizer)
#                         normlized_item = [token.strip(USTRIPPED_CHARACTERS) for token in tokenized_item]   
#                         tokenized_op, _ = string_tokenizer(op, tokenizer)
#                         normlized_tokenized_op = [token.strip(USTRIPPED_CHARACTERS) for token in tokenized_op]
#                         if set(normlized_tokenized_op) in set(normlized_item):
#                       
                        candidates = []
                        candidate = to_number(table.iloc[item[0], item[1]])
                        candidates.extend([candidate, round(100*candidate,2), -1*candidate, round(-100*candidate,2)])
                        if to_number(op) in candidates:
                            op_index.append((0,j))

                if len(paragraph_token_index_answer) > 0:
                    for j, item in enumerate(paragraph_token_index_answer):
                        normlized_item = [token.strip(USTRIPPED_CHARACTERS) for token in input_tokens[item[0]: item[1]]]
                        tokenized_op, _ = string_tokenizer(op, tokenizer)
                        normlized_tokenized_op = [token.strip(USTRIPPED_CHARACTERS) for token in tokenized_op]
                        if "".join(normlized_tokenized_op) in "".join(normlized_item):
                             op_index.append((1,j))
                
                all_op_index.append(op_index)
        except:
            import pdb
#             pdb.set_trace()
                        
        all_op_comps = list(product(*all_op_index))
        all_lfs = []
        
        for op_comp in all_op_comps:
            lf_tmp = lf.copy()
            for idx in range(len(op_comp)):
                item = op_comp[idx]
                if item[0] == 1:
                    s = paragraph_token_index_answer[item[1]][0]
                    e =  paragraph_token_index_answer[item[1]][1]
                    unit = ['VALUE(', s, e-1, ')']
                    lf_tmp = lf_tmp[:arguments[idx][0]+ 3* idx] + unit + lf_tmp[arguments[idx][0]+ 3* idx +1:]
                if item[0] == 0:
                    s = answer_table_cell_token_index[item[1]][0]
                    e =  answer_table_cell_token_index[item[1]][1]
                    unit = ['CELL_VALUE(', s, e-1, ')']
                    lf_tmp = lf_tmp[:arguments[idx][0]+ 3* idx] + unit + lf_tmp[arguments[idx][0]+ 3* idx +1:]
            all_lfs.append(lf_tmp)

        return all_lfs       
            
    
    def find_evidence_from_tags(self, tags):
        '''
        extract all the evidence
        '''
        evidence_pos = []
        len = tags.size(1)
        start = 0
        flag=False
        for i in range(len):
            if flag==False and tags[0][i]==1:
                flag=True
                start = i
                continue
            elif flag==False and tags[0][i]==0:
                continue
            elif (i==len-1 or tags[0][i]==0) and flag:
                flag=False
                evidence_pos.append((start, i))
                start = i+1
                continue
            else:
                flag=True
                continue
        return evidence_pos
                
def convert_inorder_to_postorder(grammer_number_sequence, grammer_space):
    '''
    in-order 
    '''
    stack = []
    post_grammer_number_sequence = []
    import pdb
#     pdb.set_trace()
    for i in range(len(grammer_number_sequence)):
        if grammer_number_sequence[i] not in grammer_space:
            post_grammer_number_sequence.append(grammer_number_sequence[i])
        elif grammer_number_sequence[i]== "(" or grammer_number_sequence[i]=="[":
            stack.append(grammer_number_sequence[i])
        elif grammer_number_sequence[i] in ["+", "-", "*", "/"]:
            while(len(stack)>0):
                if compare_priority(stack[-1], grammer_number_sequence[i]):
                    post_grammer_number_sequence.append(stack.pop())

                else:
                    break
            stack.append(grammer_number_sequence[i])
        else:
            ## "]" or ")"
            while(stack[-1]!="(" and stack[-1]!="["):
                post_grammer_number_sequence.append(stack.pop())
            stack.pop()
    while len(stack)>0:
        post_grammer_number_sequence.append(stack.pop())
        
    return post_grammer_number_sequence
                
def compare_priority(c1, c2):
    priority={ "*":1, "/":1, "+":0, "-":0}
    if c1 not in priority or c2 not  in priority:
        return False
    priority_value_1 = priority[c1]
    priority_value_2 = priority[c2]

    return priority_value_1>=priority_value_2

def get_operators_from_derivation(derivation, grammer_space):
    grammer_number_sequence = []
    num=""
    for idx, char in enumerate(derivation):
        if is_whitespace(char):
            if len(num)>0:
                grammer_number_sequence.append(num)
                num=""
            continue
        if char in grammer_space:
            if len(num)>0:
                grammer_number_sequence.append(num)
                num=""
            grammer_number_sequence.append(char)
        else:
            num+=char
            if idx == len(derivation)-1:
                grammer_number_sequence.append(num)
    return grammer_number_sequence


def get_lf_from_post_grammer_number_sequence(grammer_number_sequence, grammer_space):
    '''
    get the logical forms from the post-order grammer and number sequence
    '''
    lf_stack = []
    
    for grammer_number  in grammer_number_sequence:
        if grammer_number not in grammer_space:
            lf_stack.append([grammer_number])
        else:
            a = lf_stack.pop()
            b = lf_stack.pop()
            
            if grammer_number == "+":
                lf_stack.append(["SUM("] + b + a + [")"])
            elif grammer_number == "-":
                lf_stack.append(["DIFF("] + b + a + [")"])
            elif grammer_number == "*":
                lf_stack.append(["TIMES("] + b + a + [")"])
            else:
                lf_stack.append(["DIV("] + b + a + [")"])
    lf = lf_stack.pop()
    '''
    change for change_ration and avg
    '''
    lf = mapping_into_change_ratio(lf)
    lf = mapping_into_avg(lf)
    
    return lf


def normalize_grammer_number_sequence(grammer_number_sequence, grammer_space):
    '''
    deal with the negative number
    '''
    normalized_grammer_number_sequence = grammer_number_sequence.copy()
    add_grammer_count = 0
    for idx, grammer_number in enumerate(grammer_number_sequence):
        if grammer_number == "-" and idx==0 :
            normalized_grammer_number_sequence = ['0'] + normalized_grammer_number_sequence
            add_grammer_count +=1
            
        elif grammer_number == "-" and (grammer_number_sequence[idx-1]=="(" or grammer_number_sequence[idx-1]=="["):
            normalized_grammer_number_sequence = normalized_grammer_number_sequence[:idx+add_grammer_count] + ['0'] + normalized_grammer_number_sequence[idx+add_grammer_count:]
            add_grammer_count +=1
        elif grammer_number == "-" and (grammer_number_sequence[idx-1]=="*" or grammer_number_sequence[idx-1]=="/"):
#             import pdb
#             pdb.set_trace()
            normalized_grammer_number_sequence = normalized_grammer_number_sequence[:idx+add_grammer_count] + ["("] + ['0'] + normalized_grammer_number_sequence[idx+add_grammer_count:idx+add_grammer_count+2] + [")"] + normalized_grammer_number_sequence[idx+add_grammer_count+3:]
            add_grammer_count +=3
        else:
            continue
        
    return  normalized_grammer_number_sequence   
            
                
def mapping_into_change_ratio(logical_form):
    rval = []
    i = 0
    while i < len(logical_form):
        if logical_form[i] == "DIV("  and i + 6 < len(logical_form) and logical_form[i+1] == "DIFF(" and logical_form[i+4] == ')' and logical_form[i+3] == logical_form[i+5] and logical_form[i+6] == ")":
            rval.extend(['CHANGE_R(', logical_form[i+2], logical_form[i+3], ')'])
            i += 7
        else:
            rval.append(logical_form[i])
            i += 1
    return rval
    
def mapping_into_avg(logical_form):
    '''
    AVG: 
    '''
    def find_end_idx(logical_form):
        stack = []
        for i in range(len(logical_form)):
            if logical_form[i][-1] == '(':
                stack.append('(')
            elif logical_form[i][0] == ')':
                stack.pop()
                if len(stack) == 0:
                    return i

    rval = []
    i = 0
    import pdb
    
#     pdb.set_trace()
    while i < len(logical_form):
        if logical_form[i] == "DIV(":
            end_idx = i + find_end_idx(logical_form[i:])
            try:
                num = int(logical_form[end_idx-1])
            except:
                i += 1
                continue
            j = 0
            while j < num-1:
                if logical_form[i+1+j] != "SUM(":
                    break
                j += 1
            if j == num-1 and j!=0:
#                 pdb.set_trace()
                tmp = ["AVG("]
#                 for k in range(0, num):
#                     tmp.append(logical_form[i+num+k])
#                 tmp.append('[')
#                 tmp.append(logical_form[i+num+0])
#                 tmp.append(logical_form[i+num+1])
#                 for k in range(0, num-2):
#                     tmp.append(logical_form[i+num+1 + 2*(k+1)])
                import pdb
#                 pdb.set_trace()
                last_end_idx = i+num-1
                for k in range(0, num-1):
                    cur_end_idx = find_end_idx(logical_form[i+num-1-k:]) + (i+num-1-k)
                    tmp.extend(logical_form[last_end_idx+1: cur_end_idx])
                    last_end_idx = cur_end_idx
#                 tmp.append("]")
                tmp.append(")")
                logical_form = logical_form[:i] + tmp + logical_form[end_idx+1:]

        i += 1

    return logical_form


if __name__=="__main__":
    
#     derivation = "[1,496 +(-879)]/ 2 - [(-879)+ 4,764]/ 2"
#     grammer_space = grammer_space = ["[", "]", "(", ")", "+", "-", "*", "/"]
#     grammer_number_sequence = get_operators_from_derivation(derivation, grammer_space)
#     normalized_grammer_number_sequence = normalize_grammer_number_sequence(grammer_number_sequence, grammer_space)
#     post_grammer_number_sequence = convert_inorder_to_postorder(normalized_grammer_number_sequence, grammer_space)
#     lf = get_lf_from_post_grammer_number_sequence(post_grammer_number_sequence, grammer_space)

    
#     print(derivation)
#     print(grammer_number_sequence)
#     print(normalized_grammer_number_sequence)
#     print(post_grammer_number_sequence)
#     print(lf)

#     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', 'SUM(', '2.9', '2.9', ')', '2.9', ')', '3', ')', 'DIV(', 'SUM(', '2.7', '2.7', ')', '2', ')', ')', 'ED']
    s = ['ST', 'DIFF(', 'DIV(', 'SUM(', '1,496', 'DIFF(', '0', '879', ')', ')', '2', ')', 'DIV(', 'SUM(', 'DIFF(', '0', '879', ')', '4,764', ')', '2', ')', ')', 'ED']
#     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', '2.9', '2.9', ')', '2', ')', 'DIV(', 'SUM(', '2.7', '2.7', ')', '2', ')', ')', 'ED']
    print(s)
    print(mapping_into_avg(s))


# #     s = ['ST', 'DIFF(', 'DIV(', 'SUM(', '2.9', '2.9', ')', '2', ')', 'DIV(', 'SUM(', '2.7', '2.7', ')', '2', ')', ')', 'ED']
# #     print(s)
# #     print(mapping_into_avg(s))
    
# #     s= ["ST", "DIV(", "SUM(", "SUM(", "SUM(", 103.9, 103.2, ")",  102.2, ")", 111, ")", 4,  ")", "ED"]
# #     print(mapping_into_avg(s))
    
        
        
        