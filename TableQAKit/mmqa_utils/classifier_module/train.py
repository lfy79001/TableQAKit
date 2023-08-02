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
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ClassifyDataset,collate
from model import ClassifierModel
import sys
from time import gmtime, strftime
import json

device = torch.device("cuda")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
MAX_LEN = 512
TYPES = ['image', 'text', 'table', 'compose']

def get_type(type_str):
    if type_str in ['ImageQ', 'ImageListQ']:
        return 'image'
    elif type_str in ['TableQ']:
        return 'table'
    elif type_str in ['TextQ']:
        return 'text'
    else:
        return 'compose'

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

def train(epoch, tokenizer, model, loader, optimizer, scheduler, logger):
    model.train()
    averge_step = len(loader) // 20
    loss_sum, step = 0, 0
    # loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    loss_func = nn.CrossEntropyLoss()
    for i, data in enumerate(tqdm(loader)):
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
        if i % averge_step == 0:
            logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
            loss_sum, step = 0, 0

def eval(model, loader, logger, save = False):
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
                    save_file[data['qids'][j]] = {
                        "predict": TYPES[predicts[j]],
                        "target": TYPES[targets[j]]
                    }
                    if predicts[j] != targets[j]:
                        key_str = f'{TYPES[targets[j]]}->{TYPES[predicts[j]]}'
                        if key_str not in error_dict.keys():
                            error_dict[key_str] = 1
                        else:
                            error_dict[key_str] += 1
    if save:
        with open(os.path.join(ROOT_DIR, 'outputs/result', 'error.json'), 'w') as f:
            json.dump(error_dict, f, indent=4)
        with open(os.path.join(ROOT_DIR, 'outputs/result', 'classify_result.json'), 'w') as f:
            json.dump(save_file, f, indent=4)
    print(f"Total: {total}")
    return acc / total


def test(model, loader):
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
                    "predict": TYPES[predicts[j]],
                }

    with open(os.path.join(ROOT_DIR, 'outputs/result', 'classify_result_test.json'), 'w') as f:
        json.dump(save_file, f, indent=4)

    print(f"Total: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptm-type', type=str, default='deberta-large')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=7e-6)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--log-file', type=str, default='train_classifier.log')
    parser.add_argument('--n-gpu', type=int, default=1)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--ckpt-file', type=str, default='classifier.pt')
    parser.add_argument('--load-dir', type=str, default='outputs')
    parser.add_argument('--load-ckpt-file', type=str, default='classifier.pt')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--no-resume', action='store_true', default=False)
    args = parser.parse_args()
    seed = args.seed
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    model = ClassifierModel(os.path.join(ROOT_DIR,f'ptm/{args.ptm_type}'), 4, 0.2)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR,f'ptm/{args.ptm_type}'))

    if (not args.eval) and (not args.test):
        logger = create_logger("Training", log_file=os.path.join(args.output_dir, args.log_file))
        train_dataset = ClassifyDataset(tokenizer,'train')
        dev_dataset = ClassifyDataset(tokenizer,'dev')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN))
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)

        t_total = len(train_dataset) * args.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps * t_total, num_training_steps=t_total
        )

        # outputs/classifier.pt 是否存在？
        # load model
        best_acc = 0
        model_load_path = os.path.join(ROOT_DIR, args.load_dir, args.load_ckpt_file)
        if (not args.no_resume) and os.path.exists(model_load_path):
            logger.info(f"loading trained parameters from {model_load_path}")
            model.load_state_dict(torch.load(model_load_path))
            logger.info("start eval....")
            acc = eval(model, dev_loader, logger)
            logger.info(f"acc: {acc}")
            best_acc = acc

        for epoch in range(args.epochs):
            logger.info(f"Training epoch: {epoch}")
            train(epoch, tokenizer, model, train_loader, optimizer, scheduler, logger)
            logger.info("start eval....")
            acc = eval(model, dev_loader, logger)
            logger.info(f"[{epoch}/{args.epochs}] acc: {acc}")
            if acc > best_acc:
                best_acc = acc
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(args.output_dir, args.ckpt_file)
                logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)
    elif args.eval:
        logger = create_logger("Evaling", log_file=os.path.join(args.output_dir, args.log_file))
        dev_dataset = ClassifyDataset(tokenizer, 'dev')
        dev_loader = DataLoader(dev_dataset, batch_size=8, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN))
        model_load_path = os.path.join(ROOT_DIR, args.load_dir, args.load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        acc = eval(model, dev_loader, logger, save=True)
        print(f"acc: {acc}")
    elif args.test:
        logger = create_logger("Testing", log_file=os.path.join(args.output_dir, args.log_file))
        test_dataset = ClassifyDataset(tokenizer, 'test')
        test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=lambda x: collate(x, tokenizer, MAX_LEN, test=True))
        model_load_path = os.path.join(ROOT_DIR, args.load_dir, args.load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start infer....")
        test(model, test_loader)
