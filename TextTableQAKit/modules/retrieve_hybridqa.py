import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging
import os, sys, math
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
import pickle
import argparse


##  添加了AdamW和linear warm up
def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):

        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)    

class RetrieveModel(nn.Module):
    def __init__(self, bert_model):
        super(RetrieveModel, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = 768
        self.projection = FFNLayer(self.hidden_size, self.hidden_size, 1, 0.2)

    def forward(self, data):
        inputs = {"input_ids": data['input_ids'], "attention_mask": data['attention_mask']}

        cls_output = self.bert_model(**inputs)[0][:,0,:]
        # ([bs, seq_len, hidden_size], [bs, hiddensize])
        # yuyi = self.bert_model(**inputs)[1]
        # yuyi = self.bert_model(**inputs)[0][:,0,:]
        
        # cls_output.shape  (bs, hidden_size)
        # cls_output = self.bert_model(input_ids=data['input_ids'], attention_mask=data['input_mask'])[0][:,0,:]
        logits = self.projection(cls_output)
        
        # probs = logits.squeeze(-1).unsqueeze(0)
        probs = torch.softmax(logits, 0)
        return probs





class MyDataset(Dataset):
    def __init__(self, tokenizer, data): 
        super(MyDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.total_data = []
        
        
        for item in tqdm(self.data): # 看字典的元素，xxx.keys()
            
            if item['labels'].count(1) == 0:
                continue
            
            question = item['question']
            path = './Hybridqa_data/WikiTables-WithLinks'
            table_id = item['table_id']
            with open(f'{path}/tables_tok/{table_id}.json', 'r') as f:
                table = json.load(f) 
            
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            
            item_data = {}
            input_ids = [] 
            question_ids = tokenizer.encode(question)
            template = '{} is {}. '
            headers =  [aa[0] for aa in table['header']]
            
            for i, row in enumerate(table['data']):
                 
                links = []
                row_ids = []
                for j, cell in enumerate(row):
                    cell_string = template.format(headers[j], cell[0])
                    cell_tokens = tokenizer.tokenize(cell_string)
                    cell_ids = tokenizer.convert_tokens_to_ids(cell_tokens)
                    row_ids.extend(cell_ids)
                    row_ids += [tokenizer.sep_token_id]
                    
                    links.extend(cell[1])
                
                passage_ids = []
                for link in links:
                    link_tokens = tokenizer.tokenize(requested_document[link])
                    link_ids = tokenizer.convert_tokens_to_ids(link_tokens)
                    passage_ids.extend(link_ids)
                    passage_ids += [tokenizer.sep_token_id]
                
                
                row_ids = question_ids + row_ids + passage_ids
                input_ids.append(row_ids)
            
            item_data['input_ids'] = input_ids
            item_data['labels'] = item['labels']
            item_data['metadata'] = item
            self.total_data.append(item_data)
            
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self, index):
        data = self.total_data[index]
        return data
            


def my_collate(data, tokenizer, bert_max_length):
    pad_id = tokenizer.pad_token_id
    bs = len(data)
    # 按bs为1去写
    data = data[0]
    
    input_ids = data['input_ids']
    max_input_length = max([len(item) for item in input_ids])
    if max_input_length > bert_max_length:
        max_input_length = bert_max_length
    
    input_ids_new = []
    for row_ids in input_ids:
        if len(row_ids) > max_input_length:
            row_ids = row_ids[:max_input_length]
        else:
            row_ids = row_ids + [pad_id] * (max_input_length - len(row_ids))
        input_ids_new.append(row_ids)
    
    input_ids = torch.tensor(input_ids_new)
    attention_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    
    if data['labels'].count(1) != 0:
        labels = torch.tensor(data['labels']) / data['labels'].count(1)
    return {"input_ids": input_ids.cuda(), "attention_mask": attention_mask.cuda(), "labels": labels.cuda(), "metadata":data}

               


def train(epoch, tokenizer, model, loader, optimizer, scheduler, logger):
    model.train()
    averge_step = len(loader) // 5
    loss_sum, step = 0, 0
    for i, data in enumerate(tqdm(loader)):
        probs = model(data)
        
        loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        loss = loss_func(probs, data['labels'].unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        
        loss_sum += loss
        step += 1
        if i % averge_step == 0:
            logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
            loss_sum, step = 0, 0

def eval(model, loader, logger):
    model.eval()
    total, acc = 0, 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            predicts = torch.argmax(probs).cpu().item()
            gold_row = torch.where(data['labels']!=0)[0].cpu().tolist()

            total += 1
            if predicts in gold_row:
                acc += 1
    print(f"Total: {total}")
    return acc / total

def main():
    device = torch.device("cuda")
    ptm_type = 'bert-base-uncased'
    train_data_path = './Hybridqa_data/train.json'
    dev_data_path = './Hybridqa_data/dev.json'
    predict_save_path = './Hybridqa_data/test.json'

    batch_size = 1
    epoch_nums = 5
    learning_rate = 7e-6
    adam_epsilon = 1e-8
    max_grad_norm = 1
    warmup_steps = 0
    is_train = 1

    seed = 2001
    output_dir = './retrieve_new'
    load_dir = './retrieve_new'
    log_file = 'log.txt'
    ckpt_file = 'ckpt.pt'
    load_ckpt_file = 'ckpt.pt'
    dataset_name = 'hybridqa'   # hotpotqa/hybridqa
    n_gpu = torch.cuda.device_count()

    bert_max_length = 512

    notice = f'is_train={is_train}, lr={learning_rate}, epoch_num={epoch_nums}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(load_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger("Training", log_file=os.path.join(output_dir, log_file))
    
    
    logger.info(notice)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        

        
    logger.info(f"{notice}")
    logger.info(f"load_dir: {load_dir}   output_dir: {output_dir}")
    logger.info(f"loading data......from {train_data_path} and {dev_data_path}")
    
    train_data = load_data(train_data_path)
    dev_data = load_data(dev_data_path)
    logger.info(f"train data: {len(train_data)}, dev data: {len(dev_data)}")
    
    ptm_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(ptm_path)
    tokenizer = None
    if is_train:
        train_dataset = MyDataset(tokenizer, train_data[:20])
    dev_dataset = MyDataset(tokenizer, dev_data[:20])
        
        

    if is_train:
        logger.info(f"train dataset: {len(train_dataset)}")
    logger.info(f"dev dataset: {len(dev_dataset)}")

    if is_train:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: my_collate(x, tokenizer, bert_max_length))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=lambda x: my_collate(x, tokenizer, bert_max_length))
    
    
    bert_model = BertModel.from_pretrained(ptm_path)
    model = RetrieveModel(bert_model)
    model.to(device)
    best_acc = -1
    if is_train:
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
        t_total = len(train_dataset) * epoch_nums // batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps * t_total, num_training_steps=t_total
        )
        
        for epoch in range(epoch_nums):
            logger.info(f"Training epoch: {epoch}")
            train(epoch, tokenizer, model, train_loader, optimizer, scheduler, logger)
            logger.info("start eval....")
            acc = eval(model, dev_loader, logger)
            logger.info(f"acc... {acc}")
            if acc > best_acc:
                best_acc = acc
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(output_dir, ckpt_file)
                logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)
    else:
        model_load_path = os.path.join(load_dir, load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        acc = eval(model, dev_loader, logger)
            

if __name__ == '__main__':
    main()