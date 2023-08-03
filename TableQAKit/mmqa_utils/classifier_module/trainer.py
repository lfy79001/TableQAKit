import torch
import os
import math
from transformers import get_linear_schedule_with_warmup, AdamW
import argparse
import random
import numpy as np
from torch import nn
from torch.nn import functional as F
import logging
from transformers import AutoTokenizer, HfArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from time import gmtime, strftime
import json
from .dataset import ClassifyDataset,collate
from .model import ClassifierModel
from .utils import get_training_args

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
MAX_LEN = 512
TYPES = ['image', 'text', 'table', 'compose']
LABELS = {
    "image": 0,
    "text": 1,
    "table": 2,
    "compose": 3
}

def get_type(type_str):
    if type_str in ['ImageQ', 'ImageListQ']:
        return 'image'
    elif type_str in ['TableQ']:
        return 'table'
    elif type_str in ['TextQ']:
        return 'text'
    else:
        return 'compose'
    
def get_keys_by_value(dict_obj, value):
    keys = []
    for k, v in dict_obj.items():
        if v == value:
            keys.append(k)
    return keys

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

class ClassifierTrainer:
    def __init__(self, type_fn = get_type, type_map = LABELS) -> None:
        args = get_training_args()
        self.args = args
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        self.model = ClassifierModel(args.bert_model, args.num_classes, args.dropout)
        # model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.type_fn = type_fn
        self.type_map = type_map
        self.loss_fn = nn.CrossEntropyLoss() if args.loss_fn_type == 'CE' else nn.BCEWithLogitsLoss(reduction='sum')
            

    def train_epoch(self, model, loader, optimizer, scheduler, logger):
        model.train()
        averge_step = len(loader) // 20
        loss_sum, step = 0, 0
        for i, data in enumerate(tqdm(loader)):
            probs = model(data) # probs for each type
            # if sum(data['labels'][0]).cpu().item()!=1:
                # data['labels'] = F.softmax(probs + (1 - data['labels'].float()) * -1e20, dim=-1)
            loss = self.loss_fn(probs, data['labels'])
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

    def train(self, resume_from = None):
        logger = create_logger("Training", log_file=os.path.join(self.args.output_dir, self.args.log_file))
        if self.train_set == None:
            self.train_set = ClassifyDataset(tokenizer=self.tokenizer,type_fn=self.type_fn, type_map=self.type_map,
                                             data_path=self.args.data_path, split='train')
        if self.dev_set == None:
            self.dev_set = ClassifyDataset(tokenizer=self.tokenizer,type_fn=self.type_fn, type_map=self.type_map,
                                           data_path=self.args.data_path, split='dev')

        train_loader = DataLoader(self.train_set, batch_size=self.args.per_device_train_batch_size, 
                                  collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length))
        
        dev_loader = DataLoader(self.dev_set, batch_size=self.args.per_device_eval_batch_size, 
                                collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length))
        
        epochs = self.args.num_train_epochs        
        device = torch.device("cuda")
        model = self.model
        model.to(device)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        t_total = len(self.train_set) * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps * t_total, num_training_steps=t_total
        )

        # outputs/classifier.pt 是否存在？
        # load model
        best_acc = 0
        model_load_path = self.args.resume_path if resume_from is None else resume_from
        if model_load_path is not None and os.path.exists(model_load_path):
            logger.info(f"loading trained parameters from {model_load_path}")
            model.load_state_dict(torch.load(model_load_path))
            logger.info("start eval....")
            acc = self.eval(dev_loader, logger)
            logger.info(f"acc: {acc}")
            best_acc = acc

        for epoch in range(epochs):
            logger.info(f"Training epoch: {epoch}")
            self.train_epoch(model, train_loader, optimizer, scheduler, logger)
            logger.info("start eval....")
            acc = self.eval(dev_loader, logger)
            logger.info(f"[{epoch}/{epochs}] acc: {acc}")
            if acc > best_acc:
                best_acc = acc
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(self.args.output_dir, self.args.ckpt_file)
                logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)


    def eval(self, loader = None, logger=None, save = False, ckpt_path = None):
        if logger == None:
            logger = create_logger("Evaling", log_file=os.path.join(self.args.output_dir, self.args.log_file))


        if loader == None:
            if self.dev_set == None:
                self.dev_set = ClassifyDataset(tokenizer=self.tokenizer,type_fn=self.type_fn, type_map=self.type_map,
                                               data_path=self.args.data_path, split='dev')
            loader = DataLoader(self.dev_set, batch_size=self.args.per_device_eval_batch_size, 
                                    collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length))

        model = self.model
        if ckpt_path != None:
            """eval independently"""
            model_load_path = ckpt_path
            logger.info(f"loading trained parameters from {model_load_path}")
            model.load_state_dict(torch.load(model_load_path))
            logger.info("start eval....")
            device = torch.device("cuda")
            model.to(device)
        else:
            logger.info("start eval...")
            device = torch.device("cuda")
            model.to(device)
            
        model.eval()
        total, acc = 0, 0
        save_file = dict()
        error_dict = dict()
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                probs = model(data)
                predicts = torch.argmax(probs, dim=1).cpu().tolist()
                targets = data['labels'].cpu().tolist()
                total += len(targets)
                results = [1 if predicts[j] == targets[j] else 0 for j in range(len(targets))]
                acc += sum(results)
                if save:
                    for j in range(len(targets)):
                        pred = get_keys_by_value(self.type_map, predicts[j])[0]
                        target = get_keys_by_value(self.type_map, targets[j])[0]
                        save_file[data['qids'][j]] = {
                            "predict": pred,
                            "target":  target,
                        }
                        if predicts[j] != targets[j]:
                            key_str = f'{target}->{pred}'
                            if key_str not in error_dict.keys():
                                error_dict[key_str] = 1
                            else:
                                error_dict[key_str] += 1
        if save:
            with open(os.path.join('./eval_result', 'error.json'), 'w') as f:
                json.dump(error_dict, f, indent=4)
            with open(os.path.join('./eval_result', 'classify_result.json'), 'w') as f:
                json.dump(save_file, f, indent=4)
        # logger.info(f"Total golds: {total}")
        return acc / total


    def test(self, ckpt_path = None, per_device_test_batch_size = 8):
        if ckpt_path == None and self.args.ckpt_for_test==None:
            raise ValueError("ckpt_path is empty")
        
        model = self.model
        device = torch.device("cuda")
        model.to(device)
        
        logger = create_logger("Testing", log_file=os.path.join(self.args.output_dir, self.args.log_file))
        if self.test_set == None:
            self.test_set = ClassifyDataset(tokenizer=self.tokenizer,type_fn=self.type_fn, type_map=self.type_map,
                                            data_path=self.args.data_path, split='test')

        loader = DataLoader(self.test_set, batch_size=per_device_test_batch_size, 
                                 collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length, test=True))

        model_load_path = ckpt_path if ckpt_path is not None else self.args.ckpt_for_test
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start infer....")
        model.eval()
        save_file = dict()
        total = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                probs = model(data)
                predicts = torch.argmax(probs, dim=1).cpu().tolist()
                total += len(predicts)
                for j in range(len(predicts)):
                    save_file[data['qids'][j]] = {
                        "predict": get_keys_by_value(self.type_map, predicts[j])[0],
                    }

        with open(self.args.test_out_path, 'w') as f:
            json.dump(save_file, f, indent=4)

        print(f"Total: {total}")