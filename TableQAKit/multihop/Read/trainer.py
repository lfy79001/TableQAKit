import sys
sys.path.append('./')
import json
import os
import math
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict
from .utils import *
from .dataset import ReadDataset, BartReadDataset
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
import numpy as np
from pathlib import Path
from .evaluate_script1 import get_raw_scores
from .model import ReadModel


class HQATrainer:
    def __init__(self, args):
        self.args = args
    
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        self.logger = create_logger("ReadModel", log_file=os.path.join(args.output_path, 'log.txt'))
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm)
        with open(args.dev_reference, 'r') as f:
            self.reference = json.load(f)

        if args.is_train:
            self.train_data = load_json(args.train_data_path)
            self.logger.info(f"train_data: {len(self.train_data)}")
        self.dev_data = load_json(args.dev_data_path)
        self.logger.info(f"dev_data: {len(self.dev_data)}")
    @staticmethod
    def collate(self, data, **training_config):
        pass
    @staticmethod
    def train(self, data, **training_config):
        pass
    @staticmethod
    def eval(self, model, loader):
        pass
    @staticmethod
    def train_epoch(self, loader, model):
        pass
    
class GenerateTrainer(HQATrainer):
    def __init__(self, args):
        super(GenerateTrainer, self).__init__(args)
        if args.is_train:
            self.train_dataset = BartReadDataset(args, self.train_data, 'train')
        self.dev_dataset = BartReadDataset(args, self.dev_data, 'dev')
        
    def collate(self, data, **training_config):
        bs = len(data)
        max_input_length = 0
        input_ids = []
        answer_ids = []
        metadata = []
        max_input_length = max([len(item['input_ids']) for item in data])
        max_answer_length = max([len(item['answer_ids']) for item in data])
        if max_input_length > training_config['max_len']:
            max_input_length = training_config['max_len']

        for i in range(bs):
            if len(data[i]['input_ids']) > max_input_length:
                input_id = data[i]['input_ids'][:max_input_length]
            else:
                input_id = data[i]['input_ids'] + (max_input_length - len(data[i]['input_ids'])) * [self.tokenizer.pad_token_id]
            input_ids.append(input_id)


            if len(data[i]['answer_ids']) > max_answer_length:
                answer_id = data[i]['answer_ids'][:max_answer_length]
            else:
                answer_id = data[i]['answer_ids'] + (max_answer_length - len(data[i]['answer_ids'])) * [self.tokenizer.pad_token_id]
            answer_ids.append(answer_id)

            metadata.append(data[i])
        input_ids = torch.tensor(input_ids)
        input_mask = torch.where(input_ids==self.tokenizer.pad_token_id, 0, 1)

        answer_ids = torch.tensor(answer_ids)
        answer_mask = torch.where(answer_ids==self.tokenizer.pad_token_id, 0, 1)

        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "answer_ids":answer_ids.cuda(), "answer_mask":answer_mask.cuda(), "metadata":metadata}

    def train(self, **training_config):
        self.training_config = training_config
        
        train_loader = DataLoader(self.train_dataset, batch_size=training_config['train_bs'], collate_fn=lambda x: self.collate(x, **training_config))
        dev_loader = DataLoader(self.dev_dataset, batch_size=training_config['dev_bs'], collate_fn=lambda x: self.collate(x, **training_config))

        model = BartForConditionalGeneration.from_pretrained(self.args.plm)
        device = torch.device("cuda")
        model.to(device)
        
        best_acc = -1
        self.optimizer = AdamW(model.parameters(), lr=training_config['lr'], eps=training_config['eps'])
        t_total = len(self.train_dataset) * training_config['epoch_num']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0 * t_total, num_training_steps=t_total)

        for epoch in range(training_config['epoch_num']):
            self.logger.info(f"Training epoch: {epoch}")
            self.train_epoch(train_loader, model)
            self.logger.info(f"start eval....")
            total_exact, outputs = self.eval(model, dev_loader)
            scores = get_raw_scores(outputs, self.reference)
            total_exact = scores['total exact']
            self.logger.info(f"{scores}")
            self.logger.info(f"Total Exact: {total_exact}")
            if total_exact > best_acc:
                best_acc = total_exact
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(self.args.output_path, 'ckpt.pt')
                self.logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)
            
    
    def eval(self, model, loader):
        model.eval()
        total, acc = 0, 0
        outputs = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                generated_ids = model.generate(
                    input_ids = data['input_ids'],
                    attention_mask=data['input_mask'],
                    max_length = 20,
                    num_beams = 3,
                    early_stopping=True
                )
                metadata = data['metadata']
                for i, item in enumerate(generated_ids):
                    total += 1
                    output = torch.masked_select(item, item.ge(3))
                    pred_answer = self.tokenizer.decode(output)
                    answer = metadata[i]['answer-text']
                    question_id = metadata[i]['question_id']
                    if pred_answer == answer:
                        acc += 1
                    outputs.append({"question_id":question_id, "pred":pred_answer})
        self.logger.info(f"outputs num: {len(outputs)}")
        return acc / total, outputs
    
            
            
    def train_epoch(self, loader, model):
        model.train()
        averge_step = len(loader) // 12
        loss_sum, step = 0, 0
        for i, data in enumerate(tqdm(loader)):
            labels = data['answer_ids'][:, 1:].clone()
            labels[labels==self.tokenizer.pad_token_id] = -100
            outputs = model(input_ids=data['input_ids'], 
                            attention_mask=data['input_mask'], 
                            decoder_input_ids=data['answer_ids'][:, :-1],
                            labels=labels)
            loss = outputs[0]
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
    

class MRCTrainer(HQATrainer):
    def __init__(self, args):
        super(MRCTrainer, self).__init__(args)
        if args.is_train:
            self.train_dataset = ReadDataset(args, self.train_data[:100], 'train')
        self.dev_dataset = ReadDataset(args, self.dev_data[:100], 'dev')
    def collate(self, data, **training_config):
        bs = len(data)
        max_input_length = 0
        input_ids = []
        metadata = []
        start_labels, end_labels = [], []
        max_input_length = max([len(item['input_ids']) for item in data])
        if max_input_length > training_config['max_len']:
            max_input_length = training_config['max_len']
        for i in range(bs):
            if len(data[i]['input_ids']) > max_input_length:
                input_id = data[i]['input_ids'][:max_input_length]
            else:
                input_id = data[i]['input_ids'] + (max_input_length - len(data[i]['input_ids'])) * [self.tokenizer.pad_token_id]
            input_ids.append(input_id)
            start_labels.append(data[i]['start'])
            end_labels.append(data[i]['end'])
            metadata.append(data[i]['metadata'])
        input_ids = torch.tensor(input_ids)
        input_mask = torch.where(input_ids==self.tokenizer.pad_token_id, 0, 1)
        start_labels = torch.tensor(start_labels)
        end_labels = torch.tensor(end_labels)
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "start_labels":start_labels.cuda(), "end_labels":end_labels.cuda(), "metadata":metadata}

    def train(self, **training_config):
        self.training_config = training_config
        
        train_loader = DataLoader(self.train_dataset, batch_size=training_config['train_bs'], collate_fn=lambda x: self.collate(x, **training_config))
        dev_loader = DataLoader(self.dev_dataset, batch_size=training_config['dev_bs'], collate_fn=lambda x: self.collate(x, **training_config))

        bert_model = AutoModel.from_pretrained(self.args.plm)
        model = ReadModel(bert_model)
        device = torch.device("cuda")
        model.to(device)
        
        best_acc = -1
        self.optimizer = AdamW(model.parameters(), lr=training_config['lr'], eps=training_config['eps'])
        t_total = len(self.train_dataset) * training_config['epoch_num']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0 * t_total, num_training_steps=t_total)

        for epoch in range(training_config['epoch_num']):
            self.logger.info(f"Training epoch: {epoch}")
            self.train_epoch(train_loader, model)
            self.logger.info(f"start eval....")
            _, outputs = self.eval(model, dev_loader)
            scores = get_raw_scores(outputs, self.reference)
            total_exact = scores['total exact']
            self.logger.info(f"{scores}")
            self.logger.info(f"Total Exact: {total_exact}")
            if total_exact > best_acc:
                best_acc = total_exact
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(self.args.output_path, 'ckpt.pt')
                self.logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)


    def train_epoch(self, loader, model):
        model.train()
        averge_step = len(loader) // 8
        loss_sum, step = 0, 0
        for it, data in enumerate(tqdm(loader)):
            start_logits, end_logits = model(data)
            loss_func = nn.CrossEntropyLoss(reduction='sum')
            start_loss = loss_func(start_logits, data['start_labels'])
            end_loss = loss_func(end_logits, data['end_labels'])
            loss = (start_loss + end_loss) / 2

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()
            self.scheduler.step()
            loss_sum += loss
            step += 1
            if it % averge_step == 0:
                self.logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
                loss_sum, step = 0, 0

    def eval(self, model, loader):
        model.eval()
        total, acc = 0, 0
        outputs = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                start_logits, end_logits = model(data)
                start_pred = torch.argmax(F.softmax(start_logits,dim=1), dim=1).cpu().tolist()
                end_pred = torch.argmax(F.softmax(end_logits,dim=1), dim=1).cpu().tolist()
                metadata = data['metadata']
                for i in range(len(metadata)):
                    total += 1
                    question_id = metadata[i]['question_id']
                    if start_pred[i] > end_pred[i]:
                        end_pred[i] = start_pred[i]

                    answer = metadata[i]['answer-text']

                    ids_token_index = metadata[i]['ids_token_index']
                    tokens = metadata[i]['tokens']

                    try:
                        start_token_index = ids_token_index[start_pred[i]]
                        end_token_index = ids_token_index[end_pred[i]]
                    except:
                        continue
                    pred_answer = ' '.join(tokens[start_token_index:end_token_index+1])
                    if pred_answer == answer:
                        acc += 1
                    outputs.append({"question_id":question_id, "pred":pred_answer})
        print(f"outputs num: {len(outputs)}")
        return acc / total, outputs
        
    