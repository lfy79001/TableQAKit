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








# question_id, question, answer,   
class EncycDataset(Dataset):
    def __init__(self, all_data, **kwargs):
        super(EncycDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['plm'])
        self.data = []
        dataset_config = kwargs['dataset_config']
        for item in tqdm(all_data):
            data_i = {}
            data_i['id'] = item[dataset_config['id']]
            data_i['question'] = item[dataset_config['question']]
            data_i['answer'] = item[dataset_config['answer']]
            data_i['content'] = kwargs['content_func'](item)
            data_i['header'] = kwargs['header_func'](item)
            data_i['label'] = kwargs['label_func'](item)

            # if len(content[0]) == 2: # HybridQA数据集
            self.data.append(data_i)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        data_i = self.data[index]
        question_ids = self.tokenizer.encode(data_i['question'])

        header = data_i['header']
        content = data_i['content']  # content is a tuple list, each item is a tuple (cell list, links list)
        row_tmp = '{} is {} '
        input_ids = []
        if isinstance(content[0], tuple):
            for i, row in enumerate(content):
                row_ids = []
                for j, cell in enumerate(row[0]): # tokenize cell
                    if cell != '':
                        cell_desc = row_tmp.format(header[j], cell)
                        cell_toks = self.tokenizer.tokenize(cell_desc)
                        cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                        row_ids += cell_ids
                for link in row[1]:               # tokenize links
                    passage_toks = self.tokenizer.tokenize(link)
                    passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                    row_ids += passage_ids
                row_ids = question_ids + row_ids + [self.tokenizer.sep_token_id]
                input_ids.append(row_ids)
        else:
            for i, row in enumerate(content):
                row_ids = []
                for j, cell in enumerate(row):
                    if cell != '':
                        cell_desc = row_tmp.format(header[j], cell)
                        cell_toks = self.tokenizer.tokenize(cell_desc)
                        cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                        row_ids += cell_ids
                row_ids = question_ids + row_ids + [self.tokenizer.sep_token_id]
                input_ids.append(row_ids)
        return input_ids, data_i['label'], data_i
        
        

# class SpreadsheetDataset(Dataset):
#     def __init__(self):
#         pass
        
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         return feature, label
    

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

            if kwargs['dataset_name'] == 'hybridqa':
                kwargs['header_func'] = hybridqa_header
                kwargs['content_func'] = hybridqa_content
                kwargs['label_func'] = hybridqa_label
            kwargs['logger'].info('Starting load dataset')
            train_dataset = EncycDataset(dataset_data['train'][11:100], **kwargs)
            dev_dataset = EncycDataset(dataset_data['dev'][10:30], **kwargs)
            # test_dataset = EncycDataset(dataset_data['test'], **kwargs)
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

        
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "label":labels.cuda()}
        
        
        
    def train_epoch(self, loader, model):
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
            self.train_epoch(train_loader, model)
            self.kwargs['logger'].info(f"start eval....")
            acc = self.eval(model, dev_loader)
          

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='common')
    parser.add_argument('--dataset_name', type=str, default='hybridqa')
    parser.add_argument('--plm', type=str, default='/home/lfy/PTM/bert-base-uncased')
    
    
    args = parser.parse_args()
    
    kwargs = {}
    kwargs['type'] = 'common'

    kwargs['dataset_name'] = 'hybridqa'
    kwargs['plm'] = '/home/lfy/PTM/bert-base-uncased'
    kwargs['dataset_config'] = {'id': 'question_id', 'question': 'question', 'answer': 'answer-text'}
    
    # need to write a function map the table to row/columnss
    
    
    # need to generate a logger for yourself
    logger = create_logger("Training", log_file=os.path.join('../outputs', 'try.txt'))
    kwargs['logger'] = logger


    
    training_config = {}
    training_config['train_bs'] = 1
    training_config['dev_bs'] = 1
    training_config['epoch_num'] = 5
    training_config['max_len'] = 512
    training_config['lr'] = 7e-6
    training_config['eps'] = 1e-8
    
    hybridqaTrainer = HybridQATrainer(**kwargs)
    # dataset = EncycDataset(dataset_name='hybridqa')
    
    hybridqaTrainer.train(**training_config)


