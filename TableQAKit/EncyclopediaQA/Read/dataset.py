import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BartTokenizer, BartForQuestionAnswering, BartForConditionalGeneration
from transformers import AutoTokenizer, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, BertForQuestionAnswering
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging
import os, sys, math
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
import argparse



class BartReadDataset(Dataset):
    def __init__(self, args, input_data, which_dataset):
        super(BartReadDataset, self).__init__()
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
        self.tokenizer = tokenizer
        self.total_data = []
        for data in tqdm(input_data):
            path = args.WikiTables
            table_id = data['table_id']
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            answer = data['answer-text']
            
            if which_dataset == 'train':
                if len(data['row_gold']) == 0:
                    gold_row = np.argmax(np.array(data['row_pre_logit']))
                elif len(data['row_gold']) == 1:
                    gold_row = data['row_gold'][0]
                else:
                    logit = data['row_pre_logit']
                    index = np.argmax(np.array(logit)[data['row_gold']])
                    gold_row = data['row_gold'][index]
            elif which_dataset == 'dev':
                gold_row = np.argmax(np.array(data['row_pre_logit']))
                
            question_ids = self.tokenizer.encode(data['question'])

            headers = [_[0] for _ in table['header']]
            template = '{} is {}. '
            
            gold_row_data = table['data'][gold_row]
            
            links = []
            row_ids = []

            for j, cell in enumerate(gold_row_data):
                
                cell_string = template.format(headers[j], cell[0]) #cell[1]代表超链接
                cell_tokens = tokenizer.tokenize(cell_string)
                cell_ids = tokenizer.convert_tokens_to_ids(cell_tokens)
                row_ids.extend(cell_ids)
                row_ids += [tokenizer.sep_token_id] #加分隔符sep
                links.extend(cell[1])
            
            passage_ids = []
            for link in links:
                
                link_tokens = tokenizer.tokenize(requested_document[link]) #超链接
                link_ids = tokenizer.convert_tokens_to_ids(link_tokens)
                passage_ids.extend(link_ids)
                passage_ids += [tokenizer.sep_token_id]
                
            input_ids = question_ids + row_ids + passage_ids
            data['input_ids'] = input_ids
            
            answer_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(data['answer-text']))
            data['answer_ids'] = [self.tokenizer.eos_token_id] + answer_ids + [self.tokenizer.eos_token_id]

            self.total_data.append(data)
            
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self, index):
        data = self.total_data[index]
        return data
        
        

class ReadDataset(Dataset):
    def __init__(self, args, data, which_dataset):
        super(ReadDataset, self).__init__()
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
        self.tokenizer = tokenizer
        self.total_data = data
        total_data = []
        for data in tqdm(self.total_data):
            path = args.WikiTables
            table_id = data['table_id']
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)

            headers = [_[0] for _ in table['header']]
            row_tmp = '{} is {}. '
            
            answer = data['answer-text']
            if which_dataset == 'train':
                if len(data['row_gold']) == 0:
                    gold_row = np.argmax(np.array(data['row_pre_logit']))
                elif len(data['row_gold']) == 1:
                    gold_row = data['row_gold'][0]
                else:
                    logit = data['row_pre_logit']
                    index = np.argmax(np.array(logit)[data['row_gold']])
                    gold_row = data['row_gold'][index]
            elif which_dataset == 'dev':
                gold_row = np.argmax(np.array(data['row_pre_logit']))
                
            row = table['data'][gold_row]
            answer_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))
            if which_dataset == 'train':
                question_ids = self.tokenizer.encode(data['question'])
                input_ids = []
                row_ids = []
                links = []
                for j, cell in enumerate(row):
                    if cell[0] != '':
                        cell_desc = row_tmp.format(headers[j], cell[0])
                        cell_toks = self.tokenizer.tokenize(cell_desc)
                        cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                        row_ids += cell_ids
                    if cell[1] != []:
                        links += cell[1]
                links_ids = []

                for link in links:
                    passage_toks = self.tokenizer.tokenize(requested_document[link])
                    passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                    links_ids += passage_ids
                input_ids = question_ids + row_ids + links_ids + [self.tokenizer.sep_token_id]
            else:
                input_ids, row_ids, links = [], [], []
                tokens = []
                tokens_lens = []
                ids_token_index = []

                tokens.append(tokenizer.cls_token)
                input_ids.append(tokenizer.cls_token_id)
                tokens_lens.append(1)

                question_tokens, question_input_ids, question_tokens_lens = self.string_tokenizer(tokenizer, data['question'])
                tokens += question_tokens
                input_ids += question_input_ids
                tokens_lens += question_tokens_lens


                tokens += [tokenizer.sep_token]
                input_ids += [tokenizer.sep_token_id]
                tokens_lens += [1]
                row_tokens, row_input_ids, row_tokens_lens, links = [], [], [], []
                for j, cell in enumerate(row):
                    if cell[0] != '':
                        cell_desc = row_tmp.format(headers[j], cell[0])
                        cell_tokens, cell_input_ids, cell_tokens_lens = self.string_tokenizer(tokenizer, cell_desc)
                        row_tokens += cell_tokens
                        row_input_ids += cell_input_ids
                        row_tokens_lens += cell_tokens_lens
                    if cell[1] != []:
                        links += cell[1]
                
                link_tokens, link_input_ids, link_tokens_lens = [], [], []

                for link in links:
                    passage = requested_document[link]
                    passage_tokens, passage_input_ids, passage_tokens_lens = self.string_tokenizer(tokenizer, passage)
                    link_tokens += passage_tokens
                    link_input_ids += passage_input_ids
                    link_tokens_lens += passage_tokens_lens
                
                input_ids += row_input_ids + link_input_ids + [tokenizer.sep_token_id]
                tokens += row_tokens + link_tokens + [tokenizer.sep_token]
                tokens_lens += row_tokens_lens + link_tokens_lens + [1]
                for i, item in enumerate(tokens_lens):
                    ids_token_index += [i] * item
                data['tokens'] = tokens
                data['ids_token_index'] = ids_token_index
            keneng_start = np.where(np.array(input_ids)==answer_ids[0])[0].tolist()
            start, end = 0, 0
            for keneng_start_i in keneng_start:
                if input_ids[keneng_start_i:keneng_start_i+len(answer_ids)] == answer_ids:
                    start = keneng_start_i
                    end = keneng_start_i+len(answer_ids) - 1
                    break
            if which_dataset == 'train':
                if input_ids[start:end+1] != answer_ids:
                    continue
                if start > 511 or end > 511:
                    continue
            data['start'] = start
            data['end'] = end
            data['input_ids'] = input_ids
            data['metadata'] = data
            total_data.append(data)
        self.data = total_data
        
    def string_tokenizer(self, tokenizer, input_string):
        str_tokens = input_string.split(' ')
        str_input_ids = []
        str_tokens_lens = []
        for i, token in enumerate(str_tokens):
            str_input_ids_i = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            str_input_ids.extend(str_input_ids_i)
            str_tokens_lens.extend([len(str_input_ids_i)])
        return str_tokens, str_input_ids, str_tokens_lens

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        return data
        


