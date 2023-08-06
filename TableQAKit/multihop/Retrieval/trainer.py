import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .utils import *
from transformers import AutoModel, AutoTokenizer, AutoConfig, Seq2SeqTrainer, TapexTokenizer, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import argparse
from .model import Retriever
from .dataset import RetrievalDataset, ColumnRetrievalDataset
from typing import Dict, List, Tuple
from pathlib import Path



class RetrievalTrainer:
    def __init__(self, args):
        self.args = args
    
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        self.logger = create_logger("ReadModel", log_file=os.path.join(args.output_path, 'log.txt'))
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm)

        if args.is_train:
            self.train_data = load_json(args.train_data_path)
            self.logger.info(f"train_data: {len(self.train_data)}")
        self.dev_data = load_json(args.dev_data_path)
        self.logger.info(f"dev_data: {len(self.dev_data)}")
        
        if args.mode == 'row':
            if args.is_train:
                self.train_dataset = RetrievalDataset(args, self.train_data)
            self.dev_dataset = RetrievalDataset(args, self.dev_data)   
        elif args.mode == 'column':
            if args.is_train:
                self.train_dataset = ColumnRetrievalDataset(args, self.train_data[:30])
            self.dev_dataset = ColumnRetrievalDataset(args, self.dev_data[:10])   
    
    # Dataset Collator
    def collate(self, data, **training_config):
        pad_id = self.tokenizer.pad_token_id
        bs = len(data)
        # 按bs为1去写
        data = data[0]
        
        
        input_ids = data['input_ids']
        max_input_length = max([len(item) for item in input_ids])
        if max_input_length > training_config['max_len']:
            max_input_length = training_config['max_len']
        
        input_ids_new = []
        for row_ids in input_ids:
            if len(row_ids) > max_input_length:
                row_ids = row_ids[:max_input_length]
            else:
                row_ids = row_ids + [pad_id] * (max_input_length - len(row_ids))
            input_ids_new.append(row_ids)
        
        input_ids = torch.tensor(input_ids_new)
        attention_mask = torch.where(input_ids==self.tokenizer.pad_token_id, 0, 1) #忽略填充pad影响
        if data['labels'].count(1) != 0:
            labels = torch.tensor(data['labels']) / data['labels'].count(1)
        return {"input_ids": input_ids.cuda(), "attention_mask": attention_mask.cuda(), "labels": labels.cuda(), "metadata":data}

    def train(self, **training_config):
        self.training_config = training_config
        
        train_loader = DataLoader(self.train_dataset, batch_size=training_config['train_bs'], collate_fn=lambda x: self.collate(x, **training_config))
        dev_loader = DataLoader(self.dev_dataset, batch_size=training_config['dev_bs'], collate_fn=lambda x: self.collate(x, **training_config))

        bert_model = AutoModel.from_pretrained(self.args.plm)
        model = Retriever(bert_model)
        device = torch.device("cuda")
        model.to(device)
        
        best_acc = -1
        self.optimizer = AdamW(model.parameters(), lr=training_config['lr'], eps=training_config['eps'])
        t_total = len(self.train_dataset) * training_config['epoch_num']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0 * t_total, num_training_steps=t_total)

        for epoch in range(training_config['epoch_num']):
            self.logger.info(f"Training epoch: {epoch}")
            self.train_epoch(train_loader, model)
            self.logger.info("start eval....")
            acc = self.eval(model, dev_loader)
            self.logger.info(f"acc... {acc}")
            if acc > best_acc:
                best_acc = acc
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(self.args.output_path, 'ckpt.pt')
                self.logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)

        
    def train_epoch(self, loader, model):
        model.train()
        averge_step = len(loader) // 12
        loss_sum, step = 0, 0
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            loss_func = nn.BCEWithLogitsLoss(reduction='sum')
            loss = loss_func(probs, data['labels'].unsqueeze(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()
            self.scheduler.step()
            loss_sum += loss
            step += 1
            if i % averge_step == 0:
                self.logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
                loss_sum, step = 0, 0
    
    def eval(self, model, loader):
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
        
    
