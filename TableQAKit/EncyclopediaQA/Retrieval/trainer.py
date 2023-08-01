import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import *
from transformers import AutoModel, AutoTokenizer, AutoConfig, Seq2SeqTrainer, TapexTokenizer, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
from logger import create_logger
from retriever import Retriever
import torch.nn.functional as F
import numpy as np
import argparse
from dataset import EncycDataset
from abc import abstractmethod, ABC
from typing import Dict, List, Tuple




def hybridqa_content(data):
    path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
    table_id = data['table_id']
    with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
        table = json.load(f)  
    with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
        requested_document = json.load(f)
    content = []
    for i, row in enumerate(table['data']):
        # read the cell of each row, put the in a list. For special HybridQA Dataset, read the cell links.
        cells_i = [item[0] for item in row]
        links_iii = [item[1] for item in row]
        links_ii = [item for sublist in links_iii for item in sublist]
        links_i = [requested_document[link] for link in links_ii]
        content.append((cells_i, links_i))
    return content

def hybridqa_header(data):
    path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
    table_id = data['table_id']
    with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
        table = json.load(f)  
    with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
        requested_document = json.load(f)
    header = [_[0] for _ in table['header']]
    return header   

def hybridqa_label(data):
    return data['labels']

class HybridQATrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['plm'])
        if kwargs['type'] == 'common': # 使用标准的数据集
            # get the path of the dataset
            dataset_path = dataset_download(kwargs['dataset_name'])
            # load data  -> dict {'train', 'dev', 'test'}
            dataset_data = load_data(dataset_path)
            
            kwargs['logger'].info('Starting load dataset')
            train_dataset = EncycDataset(dataset_data['train'][11:100], **kwargs)
            dev_dataset = EncycDataset(dataset_data['dev'][10:30], **kwargs)
            test_dataset = EncycDataset(dataset_data['test'], **kwargs)
            kwargs['logger'].info(f"train_dataset: {len(train_dataset)}")
            kwargs['logger'].info(f"dev_dataset: {len(dev_dataset)}")
            # print(f"test_dataset: {len(test_dataset)}")
            self.train_dataset = train_dataset
            self.dev_dataset = dev_dataset     
    
    # Dataset Collator
    def collate(self, data):
        data = data[0]
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        rows_ids = data[0]
        max_input_length = max([len(i) for i in rows_ids])
        if max_input_length > 512:
            max_input_length = 512
    
        input_ids = []
        metadata = []
        for item in rows_ids:
            if len(item) > max_input_length:
                item = item[:max_input_length]
            else:
                item = item + (max_input_length - len(item)) * [pad_id]
            input_ids.append(item) 
        input_ids = torch.tensor(input_ids)
        input_mask = torch.where(input_ids==self.tokenizer.pad_token_id, 0, 1)
        labels = torch.tensor(data[1])
        metadata = data[2]

        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "label":labels.cuda(), "metadata": metadata}

    
    # Dataset Collator
    def test_collate(self, data):
        data = data[0]
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        rows_ids = data[0]
        max_input_length = max([len(i) for i in rows_ids])
        if max_input_length > 512:
            max_input_length = 512
    
        input_ids = []
        metadata = []
        for item in rows_ids:
            if len(item) > max_input_length:
                item = item[:max_input_length]
            else:
                item = item + (max_input_length - len(item)) * [pad_id]
            input_ids.append(item) 
        input_ids = torch.tensor(input_ids)
        input_mask = torch.where(input_ids==self.tokenizer.pad_token_id, 0, 1)
        metadata = data[2]

        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "metadata": metadata}

        
    def train_epoch(self, loader, model, logger):
        model.train()
        averge_step = len(loader) // 12
        loss_sum, step = 0, 0
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            loss_func = nn.BCEWithLogitsLoss(reduction='sum')
            loss = loss_func(probs, F.normalize(data['label'].float(), p=1, dim=0).unsqueeze(0))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()
            self.scheduler.step()
            loss_sum += loss
            step += 1
            if i % averge_step == 0:
                logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
                loss_sum, step = 0, 0
    
    def eval(self, model, loader):
        model.eval()
        total, acc = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                probs = model(data)
                predicts = torch.argmax(F.softmax(probs,dim=1), dim=1).cpu().tolist()
                gold_row = [np.where(item==1)[0].tolist() for item in data['label'].cpu().numpy()]
                for i in range(len(predicts)):
                    total += 1
                    if predicts[i] in gold_row[i]:
                        acc += 1
        self.kwargs['logger'].info(f"Total: {total}")
        print(f'recall: {acc / total}')
        return acc / total
        
    
    def train(self, **training_config):
        self.training_config = training_config
        
        train_loader = DataLoader(self.train_dataset, batch_size=training_config['train_bs'], collate_fn=lambda x: self.collate(x))
        dev_loader = DataLoader(self.dev_dataset, batch_size=training_config['dev_bs'], collate_fn=lambda x: self.collate(x))

        
        device = torch.device("cuda")
        bert_model = AutoModel.from_pretrained(self.kwargs['plm'])
        model = Retriever(bert_model)
        model.to(device)
        self.optimizer = AdamW(model.parameters(), lr=training_config['lr'], eps=training_config['eps'])
        t_total = len(self.train_dataset) * training_config['epoch_num']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0 * t_total, num_training_steps=t_total)
        for epoch in range(training_config['epoch_num']):
            self.kwargs['logger'].info(f"Training epoch: {epoch}")
            self.train_epoch(train_loader, model, self.kwargs['logger'])
            self.kwargs['logger'].info(f"start eval....")
            acc = self.eval(model, dev_loader)
    
    @abstractmethod 
    def convert(self) -> Tuple(List, List):
        pass
        
        