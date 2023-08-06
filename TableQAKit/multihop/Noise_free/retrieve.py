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
        self.hidden_size = self.bert_model.embeddings.word_embeddings.embedding_dim
        self.projection = FFNLayer(self.hidden_size, self.hidden_size, 1, 0.2)

    def forward(self, data):
        inputs = {"input_ids": data['input_ids'], "attention_mask": data['input_mask']}
        cls_output = self.bert_model(**inputs)[0][:,0,:]
        logits = self.projection(cls_output)
        bs = data['labels'].size(0)
        device = cls_output.device
        probs = logits.squeeze(-1).unsqueeze(0)
        probs = torch.softmax(probs, -1)
        return probs



class TypeDataset(Dataset):
    def __init__(self, tokenizer, data, is_train, data_use, rerank_link, is_test=0) -> None:
        super(TypeDataset, self).__init__()
        self.tokenizer = tokenizer
        total_data = []
        self.is_train = is_train
        self.is_test = is_test
        self.data_use = data_use  
        self.rerank_link = rerank_link   
        if data_use == 0:    
            total_data = data
        elif data_use == 1:   
            for item in data:
                if sum(item['labels']) != 0:
                    total_data.append(item)
        elif data_use == 2:   
            for item in data:
                if sum(item['labels']) == 1:
                   total_data.append(item)
        self.data = []
        # import pdb; pdb.set_trace()
        for data in tqdm(total_data):
            path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
            table_id = data['table_id']
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            dot_token = self.tokenizer.additional_special_tokens[0]
            dot_token_id = self.tokenizer.convert_tokens_to_ids(dot_token)
            question_ids = self.tokenizer.encode(data['question'])
            headers = [_[0] for _ in table['header']]
            row_tmp = '{} is {} {}'
            row_links = []
            input_ids = []
            for i, row in enumerate(table['data']):
                row_ids = []
                links = []
                for j, cell in enumerate(row):
                    if cell[0] != '':
                        cell_desc = row_tmp.format(headers[j], cell[0], dot_token)
                        cell_toks = self.tokenizer.tokenize(cell_desc)
                        cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                        row_ids += cell_ids
                    if cell[1] != []:
                        links += cell[1]
                row_links.append(links.copy())
                if rerank_link:
                    links = self.generate_new_links(links, data)
                for link in links:
                    passage_toks = self.tokenizer.tokenize(requested_document[link])
                    passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                    row_ids += passage_ids + [dot_token_id]
                row_ids = question_ids + row_ids + [self.tokenizer.sep_token_id]
                input_ids.append(row_ids)
            data['input_ids'] = input_ids
            data['row_links'] = row_links
            self.data.append(data)
    
    def generate_new_links(self, links, data):
        if self.is_train:
            answer_link = [item[2] for item in data['answer-node']]
            new_links, other_links = [], []
            for link in links:
                if link in answer_link:
                    new_links.append(link)
                else:
                    other_links.append(link)
            new_links += other_links
            return new_links
        else:
            aa = 1
            links_rank = data['links_rank']
            total_links = data['links']
            row_link_id = [total_links.index(item) for item in links]
            row_link_id_rank = [links_rank.index(item) for item in row_link_id]
            final_rank = np.argsort(row_link_id_rank).tolist()
            new_links = []
            for item in final_rank:
                new_links.append(links[item])
            return new_links
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        if not self.is_test:
            return data['input_ids'], data['labels'], data
        else:
            return data['input_ids'], [0]*len(data['input_ids']), data
            
                    
def collate(data, tokenizer, bert_max_length, is_test=0):
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    row_num = [len(item[1]) for item in data]
    max_row_num = max(row_num)
    bs = len(data)
    max_input_length = 0
    for i in range(bs):
        rows_i = data[i][0]
        max_row_length_i = max([len(i) for i in rows_i])
        if max_row_length_i > bert_max_length:
            max_input_length = bert_max_length
            break
        if max_row_length_i > max_input_length:
            max_input_length = max_row_length_i
  
    input_ids = []
    row_mask = torch.zeros(bs, max_row_num)
    labels = torch.zeros(bs, max_row_num)
    metadata = []
    for i in range(bs):
        row_mask[i][:row_num[i]] = 1
        labels[i][:len(data[i][1])] = torch.tensor(data[i][1])
        input_data_i = []
        for item in data[i][0]:
            if len(item) > bert_max_length:
                item = item[:bert_max_length]
            else:
                item = item + (max_input_length - len(item)) * [pad_id]
            input_data_i.append(item)
        input_ids.extend(input_data_i)
        metadata.append(data[i][2])       
    input_ids = torch.tensor(input_ids)
    input_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    if not is_test:
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "labels":labels.cuda(),\
            "row_mask": row_mask.cuda(), "metadata":metadata, "max_row_num": max_row_num}
    else:
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(),"labels":labels.cuda(),\
            "row_mask": row_mask.cuda(), "metadata":metadata, "max_row_num": max_row_num}


def train(epoch, tokenizer, model, loader, optimizer, scheduler, logger, is_firststage):
    model.train()
    averge_step = len(loader) // 12
    loss_sum, step = 0, 0
    for i, data in enumerate(tqdm(loader)):
        probs = model(data)
        if not is_firststage:
            if sum(data['labels'][0]).cpu().item()!=1:
                data['labels'] = F.softmax(probs + (1 - data['labels'].float()) * -1e20, dim=-1)
        loss_func = nn.BCEWithLogitsLoss(reduction='sum')
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

def eval(model, loader, logger):
    model.eval()
    total, acc = 0, 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            predicts = torch.argmax(F.softmax(probs,dim=1), dim=1).cpu().tolist()
            gold_row = [np.where(item==1)[0].tolist() for item in data['labels'].cpu().numpy()]
            score_rank = torch.argsort(-probs, dim=1).cpu().tolist()[0]
            scores, row_rank = torch.sort(probs, dim=1, descending=True)[0][0].cpu().numpy(), torch.sort(probs, dim=1, descending=True)[1][0].cpu().numpy() 
            
            data['metadata'][0]['row_links']
            question = data['metadata'][0]['question']
            row_links = data['metadata'][0]['row_links']
            links = data['metadata'][0]['links']
            links_rank = data['metadata'][0]['links_rank']
            link_labels = data['metadata'][0]['link_labels']
            labels = data['metadata'][0]['labels']

            for i in range(len(predicts)):
                total += 1
                if predicts[i] in gold_row[i]:
                    acc += 1
    print(f"Total: {total}")
    return acc / total

def eval_file(model, loader, logger):
    model.eval()
    total, acc = 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            predicts = torch.argmax(F.softmax(probs,dim=1), dim=1).cpu().tolist()
            predcits_logits = probs.cpu().detach().tolist()
            gold_row = [np.where(item==1)[0].tolist() for item in data['labels'].cpu().numpy()]
            metadatas = data['metadata']
            for j, metadata in enumerate(metadatas):
                metadata['row_pre'] = predicts[j]
                metadata['row_gold'] = gold_row[j]
                metadata['row_pre_logit'] = predcits_logits[j]
                outputs.append(metadata)
            for i in range(len(predicts)):
                total += 1
                if predicts[i] in gold_row[i]:
                    acc += 1

    return acc / total, outputs

def test_file(model, loader, logger):
    model.eval()
    total, acc = 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            predicts = torch.argmax(F.softmax(probs,dim=1), dim=1).cpu().tolist()
            predcits_logits = probs.cpu().detach().tolist()
            metadatas = data['metadata']
            for j, metadata in enumerate(metadatas):
                metadata['row_pre'] = predicts[j]
                metadata['row_pre_logit'] = predcits_logits[j]
                metadata.pop('input_ids')
                outputs.append(metadata)
    return outputs

def read_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def main():
    device = torch.device("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptm_type', type=str, default='bert-base', help='Pre-trained model to use')
    parser.add_argument('--train_data_path', type=str, default='./Data/HybridQA/train.p.json', help='Path to training data')
    parser.add_argument('--dev_data_path', type=str, default='./Data/HybridQA/dev.p.json', help='Path to development data')
    parser.add_argument('--predict_save_path', type=str, default='./Data/HybridQA/dev.row.json', help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epoch_nums', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=7e-6, help='Learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--is_train', type=int, default=1, help='Whether to train the model')
    parser.add_argument('--is_test', type=int, default=0, help='Whether to test the model')
    parser.add_argument('--is_generate', type=int, default=0, help='Whether to generate predictions')
    parser.add_argument('--is_firststage', type=int, default=1, help='Whether to perform first-stage retrieval')
    parser.add_argument('--rerank_link', type=int, default=1, help='Whether to rerank links')
    parser.add_argument('--seed', type=int, default=2001, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./retrieve1', help='Output directory')
    parser.add_argument('--load_dir', type=str, default='./retrieve1', help='Directory to load model from')
    parser.add_argument('--bert_max_length', type=int, default=512, help='Maximum length of input sequence for BERT')
    parser.add_argument('--config_file', type=str, default='config_retrieve1.json')

    args = parser.parse_args()
    
    training_config = read_config(args.config_file)

    ptm_type = args.ptm_type
    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path
    predict_save_path = args.predict_save_path
    warmup_steps = args.warmup_steps
    is_train = args.is_train
    is_test = args.is_test
    is_generate = args.is_generate
    is_firststage = args.is_firststage
    rerank_link = args.rerank_link
    seed = args.seed
    output_dir = args.output_dir
    load_dir = args.load_dir
    bert_max_length = args.bert_max_length
    
    batch_size = training_config['batch_size']
    epoch_nums = training_config['epoch_nums']
    bert_max_length = training_config['bert_max_length']
    learning_rate = training_config['learning_rate']
    adam_epsilon = training_config['adam_epsilon']



    log_file = 'log.txt'
    ckpt_file = 'ckpt.pt'
    load_ckpt_file = 'ckpt.pt'
    n_gpu = torch.cuda.device_count()
    notice = f'is_train={is_train}, is_test={is_test}, is_firststage={is_firststage}, lr={learning_rate}, epoch_num={epoch_nums}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(load_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger("Training", log_file=os.path.join(output_dir, log_file))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        

        
    logger.info(f"{notice}")
    logger.info(f"load_dir: {load_dir}   output_dir: {output_dir}")
    logger.info(f"loading data......from {train_data_path} and {dev_data_path}")
    train_data, dev_data = load_data(train_data_path), load_data(dev_data_path)
    logger.info(f"train data: {len(train_data)}, dev data: {len(dev_data)}")
    
    if ptm_type == 'bert-large':
        ptm_path = './PTM/bert-large-uncased'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = BertTokenizer.from_pretrained(ptm_path)
        bert_model = BertModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))
    elif ptm_type == 'bert-base':
        ptm_path = './PTM/bert-base-uncased'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = BertTokenizer.from_pretrained(ptm_path)
        bert_model = BertModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))
    elif ptm_type == 'deberta':
        ptm_path = './PTM/deberta-base'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = AutoTokenizer.from_pretrained(ptm_path)
        bert_model = AutoModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))

    if is_firststage:
        if is_train:
            train_dataset = TypeDataset(tokenizer, train_data, is_train=1, data_use=2, rerank_link=rerank_link)
        dev_dataset = TypeDataset(tokenizer, dev_data, is_train=0, data_use=2, rerank_link=rerank_link, is_test=is_test)
    else:
        if is_train:
            train_dataset = TypeDataset(tokenizer, train_data, 1, 1, rerank_link)
        dev_dataset = TypeDataset(tokenizer, dev_data, 0, 0, rerank_link, is_test)


    if is_train:
        logger.info(f"train dataset: {len(train_dataset)}")
    logger.info(f"dev dataset: {len(dev_dataset)}")

    if is_train:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, bert_max_length, is_test))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, bert_max_length, is_test))
    
    model = RetrieveModel(bert_model)
    model.to(device)

    if not is_firststage and is_train:
        model_load_path = os.path.join(load_dir, load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))

    best_acc = 0
    if is_train:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
        t_total = len(train_dataset) * epoch_nums
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps * t_total, num_training_steps=t_total
        )

        for epoch in range(epoch_nums):
            logger.info(f"Training epoch: {epoch}")
            train(epoch, tokenizer, model, train_loader, optimizer, scheduler, logger, is_firststage)
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
        if is_generate:
            if not is_test:
                acc, outputs = eval_file(model, dev_loader, logger)
                print(f"acc: {acc}")
            else:
                outputs = test_file(model, dev_loader, logger)
            with open(predict_save_path, 'w') as f:
                json.dump(outputs, f, indent=2)
            logger.info(f"saving to {predict_save_path}")
            
        else:
            acc = eval(model, dev_loader, logger)
            print(f"acc: {acc}")
    

if __name__ == '__main__':
    main()