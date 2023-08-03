import torch
import os
import math
from transformers import get_linear_schedule_with_warmup
import argparse
import random
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from time import gmtime, strftime
import json
from .utils import get_training_args
from .dataset import RetrieverDataset,collate
from .model import RetrieverModel

device = torch.device("cuda")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
MAX_LEN = 512


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


class RetrieverTrainer:

    def __init__(self, caption_file_name = 'mmqa_captions_llava.json') -> None:
        args = get_training_args()
        self.args = args
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        self.model = RetrieverModel(bert_model=args.bert_model, dropout=args.dropout) 
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.caption_file_name = caption_file_name
        self.loss_fn = nn.CrossEntropyLoss() if args.loss_fn_type == 'CE' else nn.BCEWithLogitsLoss(reduction='sum')

    def train(self, resume_from = None):
        logger = create_logger("Training", log_file=os.path.join(self.args.output_dir, self.args.log_file))

        if self.train_set == None:
            self.train_set = RetrieverDataset(tokenizer=self.tokenizer, data_path=self.args.data_path,
                                         split='train', caption_file_name=self.caption_file_name)
        if self.dev_set == None:
            self.dev_set = RetrieverDataset(tokenizer=self.tokenizer, data_path=self.args.data_path, 
                                       split='dev', caption_file_name=self.caption_file_name)
        
        train_loader = DataLoader(self.train_set, batch_size=self.args.per_device_train_batch_size, 
                                  collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length, self.args.image_or_text),shuffle=True)
        dev_loader = DataLoader(self.dev_set, batch_size=self.args.per_device_eval_batch_size, 
                                collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length, self.args.image_or_text))
        
        epochs = self.args.num_train_epochs
        device = torch.device("cuda")
        model = self.model
        model.to(device)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate)

        t_total = len(self.train_set) * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps * t_total, num_training_steps=t_total
        )

        # outputs/retriever.pt 是否存在？
        # load model
        best_recall = 0
        model_load_path = self.args.resume_path if resume_from is None else resume_from
        if model_load_path is not None and os.path.exists(model_load_path):
            logger.info(f"loading trained parameters from {model_load_path}")
            model.load_state_dict(torch.load(model_load_path))
            logger.info("start eval....")
            recall = self.eval(dev_loader, logger)
            logger.info(f"recall: {recall}")
            best_recall = recall

        for epoch in range(epochs):
            logger.info(f"Training epoch: {epoch}")
            # try: 
            model.train()
            # averge_step = len(train_loader) // 50
            # eval_step = len(train_loader) // 8 # 1/8 个 epoch eval 一次
            logging_steps = self.args.logging_steps
            eval_steps = self.args.eval_steps
            loss_sum, step = 0, 0
            loss_func = self.loss_fn

            for i, data in enumerate(tqdm(train_loader)):
                if len(data['metadata']['docs']) > 0:
                    probs = model(data) # probs for each type
                    # if sum(data['labels'][0]).cpu().item()!=1:
                        # data['labels'] = F.softmax(probs + (1 - data['labels'].float()) * -1e20, dim=-1)

                    loss = loss_func(probs, data['labels'])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()
                    loss_sum += loss
                    step += 1
                if (i+1) % logging_steps == 0:
                    logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
                    loss_sum, step = 0, 0

                if (i+1) % eval_steps == 0:
                    best_recall = self.eval_process(dev_loader, logger, best_recall)
            # except KeyboardInterrupt:
            #     logger.info("Early stop!")
            #     break
                        
            best_recall = self.eval_process(dev_loader, logger, best_recall)

    def eval(self, loader = None, logger = None, save_type = None, ckpt_path = None):
        
        if logger == None:
            logger = create_logger("Evaling", log_file=os.path.join(self.args.output_dir, self.args.log_file))

        if loader == None:
            if self.dev_set == None:
                self.dev_set = RetrieverDataset(tokenizer=self.tokenizer, data_path=self.args.data_path, 
                                       split='dev', caption_file_name=self.caption_file_name)
            loader = DataLoader(self.dev_set, batch_size=self.args.per_device_eval_batch_size, 
                                collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length, self.args.image_or_text))
        model = self.model

        if ckpt_path!=None:
            model_load_path = ckpt_path
            logger.info(f"loading trained parameters from {model_load_path}")
            model.load_state_dict(torch.load(model_load_path))
            logger.info("start eval....")
            device = torch.device("cuda")
            model.to(device)
        else:
            logger.info("start eval....")
            device = torch.device("cuda")
            model.to(device)

        model.eval()
        total_gold = 0
        recall_num = 0
        scores_dict = dict()
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                if len(data['metadata']['docs']) > 0:
                    probs = model(data)
                    metadata = data['metadata']
                
                    pb_n = probs.cpu().numpy()
                    if save_type is not None:
                        scores_dict[str(i)] = dict()
                        for idx, doc_id in enumerate(metadata['docs']):
                            scores_dict[str(i)][doc_id] = float(pb_n[idx])
    
                    # 取 top-n
                    if probs.shape[0] > self.args.top_n:
                        predicts = torch.topk(probs, self.args.top_n, dim=-1).indices.cpu().numpy() # indices of preds
                    else:
                        predicts = torch.topk(probs, probs.shape[0], dim=-1).indices.cpu().numpy()
                    
                    retrieve_ids = [metadata['docs'][predicts[i]] for i in range(len(predicts))] # id of retrieved
                    targets = [metadata['gold_docs'][i] for i in range(len(metadata['gold_docs']))] # id of gold
                    total_gold += len(targets)
                    recall_num += sum([1 if targets[i] in retrieve_ids else 0 for i in range(len(targets))])
                else:
                    if save_type is not None:
                        scores_dict[str(i)] = dict()
    
        if save_type is not None:
            save_path = f'eval_result/retrieve_{save_type}.json'
            with open(save_path, 'w') as f:
                json.dump(scores_dict, f)

        logger.info(f"Total: {total_gold}\nRecall: {recall_num}")
        return recall_num / total_gold
    
    
    def eval_process(self, dev_loader, logger, best_recall):
        recall = self.eval(dev_loader, logger)
        model = self.model
        if recall > best_recall:
            best_recall = recall
            logger.info(f"best recall: {recall}")
            model_to_save = model.module if hasattr(model, "module") else model
            model_save_path = os.path.join(self.args.output_dir, f'{self.args.ckpt_file}_{self.args.image_or_text}.pt')
            logger.info(f"saving model...to {model_save_path}")
            torch.save(model_to_save.state_dict(), model_save_path)
        return best_recall

    def test(self, ckpt_path = None,save_type = None, per_device_test_batch_size = 1):
        if ckpt_path == None and self.args.ckpt_for_test==None:
            raise ValueError("ckpt_path is empty")
    
        logger = create_logger("Testing", log_file=os.path.join(self.args.output_dir, self.args.log_file))
        if self.test_set == None:
            self.test_set = RetrieverDataset(tokenizer=self.tokenizer, data_path=self.args.data_path, 
                                       split='test', caption_file_name=self.caption_file_name)
        loader = DataLoader(self.test_set, batch_size=per_device_test_batch_size, 
                                 collate_fn=lambda x: collate(x, self.tokenizer, self.args.max_length, image_or_text=self.args.image_or_text, test=True))
        model = self.model
        model_load_path = ckpt_path if ckpt_path is not None else self.args.ckpt_for_test
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start infer....")

        device = torch.device("cuda")
        model.to(device)
        model.eval()
        scores_dict = dict()
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                if len(data['metadata']['docs']) > 0:
                    probs = model(data)
                    metadata = data['metadata']
                    pb_n = probs.cpu().numpy()
                    if save_type is not None:
                        scores_dict[str(i)] = dict()
                        for idx, doc_id in enumerate(metadata['docs']):
                            scores_dict[str(i)][doc_id] = float(pb_n[idx])
                else:
                    if save_type is not None:
                        scores_dict[str(i)] = dict()

        with open(self.args.test_out_path, 'w') as f:
            json.dump(scores_dict, f, indent=4)