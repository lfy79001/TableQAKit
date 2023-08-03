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
from dataset import RetrieverDataset,collate
from model import RetrieverModel
import sys
from time import gmtime, strftime
import json

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

def eval(model, loader, logger, save_type = None):
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

                # 取 top3
                if probs.shape[0] > 3:
                    predicts = torch.topk(probs, 3, dim=-1).indices.cpu().numpy() # indices of preds
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
        save_path = os.path.join(ROOT_DIR, f"outputs/retrieve_{save_type}.json")
        with open(save_path, 'w') as f:
            json.dump(scores_dict, f)

    print(f"Total: {total_gold}\nRecall: {recall_num}")
    return recall_num / total_gold


def eval_process(model, dev_loader, logger, args, best_recall):
    logger.info("start eval....")
    recall = eval(model, dev_loader, logger)
    logger.info(f"recall: {recall}")
    if recall > best_recall:
        best_recall = recall
        model_to_save = model.module if hasattr(model, "module") else model
        model_save_path = os.path.join(args.output_dir, f'{args.ckpt_file}_{args.image_or_text}.pt')
        logger.info(f"saving model...to {model_save_path}")
        torch.save(model_to_save.state_dict(), model_save_path)

    return best_recall

def test(model, loader, save_type = None):
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

    if save_type is not None:
        save_path = os.path.join(ROOT_DIR, f"outputs/retrieve_test_{save_type}.json")
        with open(save_path, 'w') as f:
            json.dump(scores_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptm-type', type=str, default='deberta-large')
    parser.add_argument('--batch-size', type=int, default=1) # only support 1
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--log-file', type=str, default='train.log')
    parser.add_argument('--n-gpu', type=int, default=1)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--ckpt-file', type=str, default='retriever')
    parser.add_argument('--load-dir', type=str, default='outputs')
    parser.add_argument('--load-ckpt-file', type=str, default='retriever')
    parser.add_argument('--no-resume', action='store_true', default=False)
    parser.add_argument('--image_or_text', type=str, default="text")
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    seed = args.seed
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    model = RetrieverModel(os.path.join(ROOT_DIR,f'ptm/{args.ptm_type}'))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR,f'ptm/{args.ptm_type}'))

    if (not args.eval) and (not args.test):
        logger = create_logger("Training", log_file=os.path.join(args.output_dir, args.log_file))

        train_dataset = RetrieverDataset(tokenizer,'train')
        dev_dataset = RetrieverDataset(tokenizer,'dev')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN, args.image_or_text),shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN, args.image_or_text))

        # optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        t_total = len(train_dataset) * args.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps * t_total, num_training_steps=t_total
        )

        # outputs/retriever.pt 是否存在？
        # load model
        best_recall = 0
        model_load_path = os.path.join(ROOT_DIR, args.load_dir, f'{args.load_ckpt_file}_{args.image_or_text}.pt')
        if (not args.no_resume) and os.path.exists(model_load_path):
            logger.info(f"loading trained parameters from {model_load_path}")
            model.load_state_dict(torch.load(model_load_path))
            logger.info("start eval....")
            recall = eval(model, dev_loader, logger)
            logger.info(f"recall: {recall}")
            best_recall = recall
            
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        for epoch in range(args.epochs):
            logger.info(f"Training epoch: {epoch}")
            try: 
                model.train()
                averge_step = len(train_loader) // 50
                eval_step = len(train_loader) // 8 # 1/8 个 epoch eval 一次
                loss_sum, step = 0, 0
                # loss_func = nn.BCEWithLogitsLoss(reduction='sum')
                loss_func = nn.CrossEntropyLoss()
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
                    if (i+1) % averge_step == 0:
                        logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
                        loss_sum, step = 0, 0

                    if (i+1) % eval_step == 0:
                        best_recall = eval_process(model, dev_loader, logger, args, best_recall)
            except KeyboardInterrupt:
                logger.info("Early stop!")
                break
                        
            best_recall = eval_process(model, dev_loader, logger, args, best_recall)
    elif args.eval:
        logger = create_logger("Evaling", log_file=os.path.join(args.output_dir, args.log_file))
        dev_dataset = RetrieverDataset(tokenizer,'dev')
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN, args.image_or_text))
        model_load_path = os.path.join(ROOT_DIR, args.load_dir, f'{args.load_ckpt_file}_{args.image_or_text}.pt')
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        recall = eval(model, dev_loader, logger, save_type = args.image_or_text)
        print(f"recall: {recall}")
    elif args.test:
        logger = create_logger("Testing", log_file=os.path.join(args.output_dir, args.log_file))
        test_dataset = RetrieverDataset(tokenizer, 'test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN, test=True))
        model_load_path = os.path.join(ROOT_DIR, args.load_dir, f'{args.load_ckpt_file}_{args.image_or_text}.pt')
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start infer....")
        test(model, test_loader, save_type = args.image_or_text)