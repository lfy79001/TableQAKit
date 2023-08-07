import sys
sys.path.append('/export/users/zhouyongwei/UniRPG/semantic-parsing-tat-qa_scale_head_backup_metric_vm')
from tag_op.data.operation_util import GRAMMER_CLASS, GRAMMER_ID, AUX_NUM, SCALECLASS, SCALE
import json
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import os

class TATExecutor(object):
    
    def __init__(self, GRAMMER_CLASS, GRAMMER_ID, AUX_NUM):
        self.GRAMMER_CLASS = GRAMMER_CLASS
        self.GRAMMER_ID = GRAMMER_ID
        self.AUX_NUM = AUX_NUM
        self.grammar_stack = []
    
    def reset(self):
        self.grammar_stack.clear()
        
    def execute(self, prog, example):
        '''
        responsible for executing a legal logical forms into answer based on the given
        knowledge form table and text
        
        param: prog: the decoded logical forms
        
        '''
        
        table_cell_tokens = example["table_cell_tokens"]
        table_cell_numbers = example["table_cell_number_value"]
        table_cell_index = example["table_cell_index"][0]
        paragraph_tokens = example["paragraph_tokens"]
        paragraph_numbers = example["paragraph_number_value"]
        paragraph_index = example["paragraph_index"][0]
        question_id = example["question_id"]
        
        import pdb
#         pdb.set_trace()
        
        if len(prog) == 0:
            return ''
        
        predict_answer = []
        GRAMMER = [key for key in self.GRAMMER_CLASS.keys()]
        try:
            for cur_prog_token in prog:
                if cur_prog_token == '<s>':
                    continue
                if cur_prog_token == '</s>':
                    break
#                 if cur_prog_token in GRAMMER[:-1] or cur_prog_token in SCALECLASS:
#                     self.grammar_stack.append([cur_prog_token])
                if cur_prog_token in GRAMMER[:-1]:
                    self.grammar_stack.append([cur_prog_token])
                elif cur_prog_token == ')':

#                     if self.grammar_stack[-1][0] in SCALECLASS:
#                         import pdb
# #                         pdb.set_trace()
#                         if self.grammar_stack[-1][0] == "PERCENT(" and ( "DIV(" in prog or "CHANGE_R(" in prog):
#                             cur_answer = np.around(self.grammar_stack[-1][1]*100, 4)
#                         else:
#                             cur_answer = self.grammar_stack[-1][1]
#                         predict_answer.append(cur_answer)
#                         break
                        
                        
                    if self.grammar_stack[-1][0] == "SPAN(":
                        start, end = int(self.grammar_stack[-1][1]), int(self.grammar_stack[-1][2]) ## token index -> a passage span
                        word_start, word_end =  int(paragraph_index[start]-1), int(paragraph_index[end]-1)
                        cur_answer = ' '.join(paragraph_tokens[word_start:word_end+1])
                        cur_answer = cur_answer.replace(" - ", "-")

                    elif self.grammar_stack[-1][0] == "CELL(":
                        start, end  = int(self.grammar_stack[-1][1]), int(self.grammar_stack[-1][2]) ## token index -> a table cell
                        cell_start, cell_end = int(table_cell_index[start]-1), int(table_cell_index[end]-1)
    #                     assert cell_start == cell_end
                        cur_answer = str(table_cell_tokens[cell_start])
                        
                    elif self.grammar_stack[-1][0] == 'COUNT(':
                        cur_answer  = len(self.grammar_stack[-1]) -1    ## count number

                    elif self.grammar_stack[-1][0] == "KEY_VALUE(":    ## key_value -> dict
                         cur_answer = {'key': self.grammar_stack[-1][1], 'value': self.grammar_stack[-1][2]}

                    elif  self.grammar_stack[-1][0] == 'VALUE(':
                        start, end = int(self.grammar_stack[-1][1]), int(self.grammar_stack[-1][2]) ## token index -> value
                        word_start, word_end =  int(paragraph_index[start]-1), int(paragraph_index[end]-1)
                        cur_answer = paragraph_numbers[word_start].item()
                        assert np.isnan(cur_answer) is not True
                        cur_answer =  np.around(cur_answer, 4)
                        if cur_answer<0:
                            cur_answer=-1*cur_answer

                    elif self.grammar_stack[-1][0] == 'CELL_VALUE(':    ## token index -> a number in cell
                        start, end = int(self.grammar_stack[-1][1]), int(self.grammar_stack[-1][2])
                        word_start, word_end =  int(table_cell_index[start]-1), int(table_cell_index[end]-1)
                        cur_answer = table_cell_numbers[word_start].item()
                        
                        if cur_answer<0:
                            cur_answer=-1*cur_answer
                            
                        assert np.isnan(cur_answer) is not True
                        cur_answer =  np.around(cur_answer, 4)

                    elif  self.grammar_stack[-1][0] == 'ARGMAX(':      ## sort
                          cur_answer = sorted(self.grammar_stack[-1][1:], key=self.get_output_key)[-1]['key']

                    elif  self.grammar_stack[-1][0] == 'ARGMIN(':  ## sort
                         cur_answer = sorted(self.grammar_stack[-1][1:], key=self.get_output_key)[0]['key']

                    elif  self.grammar_stack[-1][0] == 'SUM(':   ## sum all the number 
                        cur_answer = 0
                        for num_value in self.grammar_stack[-1][1:]:
                            cur_answer += num_value
                        cur_answer =  np.around(cur_answer, 4)

                    elif self.grammar_stack[-1][0] == 'DIFF(':  ## diff 
                        cur_answer = self.grammar_stack[-1][1] - self.grammar_stack[-1][2]
                        cur_answer =  np.around(cur_answer, 4)

                    elif self.grammar_stack[-1][0] == 'DIV(':   ## div
                        if abs(self.grammar_stack[-1][2]) >= 1e-8:
                            cur_answer = self.grammar_stack[-1][1] / self.grammar_stack[-1][2]
                        else:
                            cur_answer = 0
                        cur_answer =  np.around(cur_answer, 4)

                    elif self.grammar_stack[-1][0] == 'AVG(':  ## 
                        cur_answer = 0
                        for num_value in self.grammar_stack[-1][1:]:
                            cur_answer += num_value
                        cur_answer = cur_answer/ len(self.grammar_stack[-1][1:])
                        cur_answer =  np.around(cur_answer, 4)

                    elif self.grammar_stack[-1][0] == 'CHANGE_R(':
                        cur_answer = self.grammar_stack[-1][1] / self.grammar_stack[-1][2] -1
                        cur_answer =  np.around(cur_answer, 4)

                    elif self.grammar_stack[-1][0] == 'MULTI-SPAN(':
                        cur_answer = []
                        for span in self.grammar_stack[-1][1:]:
                            cur_answer.append(span)

                    elif self.grammar_stack[-1][0] == 'TIMES(':
                        assert self.grammar_stack[-1][0] == 'TIMES('
                        cur_answer = 1
                        for num_value in self.grammar_stack[-1][1:]:
                            cur_answer *= num_value
                        cur_answer =  np.around(cur_answer, 4)

                    self.grammar_stack.pop()
                    
#                     if isinstance(cur_answer, np.int64):
#                         cur_answer = int(cur_answer)
#                     self.grammar_stack[-1].append(cur_answer)
                    if len(self.grammar_stack) == 0:
                        predict_answer.append(cur_answer)
                    else:
                        self.grammar_stack[-1].append(cur_answer)

                else:   ## a index  or a constant
                    if self.grammar_stack[-1][0] in ['SPAN(', 'CELL(', 'VALUE(', 'CELL_VALUE(']:
                        self.grammar_stack[-1].append(cur_prog_token)
                    else:
                        assert cur_prog_token in self.AUX_NUM
                        self.grammar_stack[-1].append(int(cur_prog_token))
        except:
            import pdb
#             pdb.set_trace()
        
    
        assert isinstance(predict_answer, list)
        return predict_answer
        
    
    def get_output_key(self,output):
        return output["value"]
    
    
    def delete_extra_zero(self, item):
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            item=str(item).rstrip('0')
            if item.endswith('.'):
                item = int(item.rstrip('.')) 
            else:
                float(item)
     
        return item
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= "")
    parser.add_argument('--logical_form_path', type=str, default= "")
#     parser.add_argument('--mode', type=str, default="groundtruth", help="predict or groundtruth")
    parser.add_argument('--mode', type=str, default="dev")
    
    args= parser.parse_args()
    
    with open(args.data_path, 'rb') as f:
        all_data = pickle.load(f)
    f.close()
    with open(args.logical_form_path, 'r') as f:
        all_logical_forms = json.load(f)
    f.close()
    executor = TATExecutor( GRAMMER_CLASS, GRAMMER_ID, AUX_NUM)
    skip = 0
    
    all_predict_result = {}
    for example in tqdm(all_data):
        question_id = example['question_id']
        predict_result = [""]
        scale = example['answer_dict']['scale']
        if question_id in all_logical_forms.keys():
#             if args.mode == 'groundtruth':
#                 prog = all_logical_forms[question_id]["groundtruth_grammar_result"]
#                 prog = prog.split(' ')
#             else:
            prog =  all_logical_forms[question_id]['predict_grammar']
            prog = prog.split(' ')
            
            predict_result = executor.execute(prog, example)
            predict_scale_label = all_logical_forms[question_id]['pred_scale']
            predict_scale = SCALE[predict_scale_label]
            
            if len(predict_result)>0 and isinstance(predict_result[0], list) :
                predict_result = [item for item in predict_result[0]]
             
            try:
                if predict_scale == "percent" and ( "DIV(" in prog or "CHANGE_R(" in prog):
                    predict_result[0] = np.around(predict_result[0]*100, 4)
            except:
                predict_result=[""]
                
        else:
            skip +=1
            
        predict_result = [str(executor.delete_extra_zero(item)) for item in predict_result]
        executor.reset()
        all_predict_result.update({
            question_id:[
                predict_result, 
                predict_scale
            ]
        })

    with open(os.path.join(os.path.dirname(args.logical_form_path), 'pred_result_on_{}.json'.format(args.mode)), 'w') as f:
        json.dump(all_predict_result, f, indent=4)
        print('finish write the predict result')
    f.close()

        

        


    
if __name__=="__main__":
    main()
    

   
    
    
