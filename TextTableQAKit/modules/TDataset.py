import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import *
from transformers import AutoTokenizer
from tqdm import tqdm
from logger import create_logger

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
        question_ids = self.tokenizer.tokenize(data_i['question'])

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

    
class DatasetManager:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if kwargs['type'] == 'common': # 使用标准的数据集
            # get the path of the dataset
            dataset_path = dataset_download(kwargs['dataset_name'])
            # load data  -> dict {'train', 'dev', 'test'}
            dataset_data = load_data(dataset_path)
            if kwargs['table_type'] == 'encyc':
                if kwargs['dataset_name'] == 'hybridqa':
                    kwargs['header_func'] = hybridqa_header
                    kwargs['content_func'] = hybridqa_content
                    kwargs['label_func'] = hybridqa_label
                kwargs['logger'].info('Starting load dataset')
                train_dataset = EncycDataset(dataset_data['train'], **kwargs)
                dev_dataset = EncycDataset(dataset_data['dev'], **kwargs)
                # test_dataset = EncycDataset(dataset_data['test'], **kwargs)
                kwargs['logger'].info(f"train_dataset: {len(train_dataset)}")
                kwargs['logger'].info(f"dev_dataset: {len(dev_dataset)}")
                # print(f"test_dataset: {len(test_dataset)}")
                self.train_dataset = train_dataset
                self.dev_dataset = dev_dataset
                pass
            elif kwargs['table_type'] == 'spreadsheet':
                pass
            elif kwargs['table_type'] == 'structured':
                pass
    
    def collate(data, **kwargs):
               
    
    def train():
        if kwargs['table_type'] == 'encyc':
            train_loader = DataLoader(self.train_dataset, batch_size=kwargs['train_bs'], collate_fn=lambda x: collate(x, **self.kwargs))
            dev_loader = DataLoader(self.dev_dataset, batch_size=kwargs['dev_bs'], collate_fn=lambda x: collate(x, **self.kwargs))
        else:        
            pass
    
    
    
if __name__ == '__main__':
    kwargs = {}
    kwargs['type'] = 'common'
    kwargs['table_type'] = 'encyc'
    kwargs['dataset_name'] = 'hybridqa'
    kwargs['plm'] = '/home/lfy/PTM/bert-base-uncased'
    kwargs['dataset_config'] = {'id': 'question_id', 'question': 'question', 'answer': 'answer-text'}
    
    # need to write a function map the table to row
    
    # need to generate a logger for yourself
    logger = create_logger("Training", log_file=os.path.join('../outputs', 'try.txt'))
    kwargs['logger'] = logger

    
    
    datasetManager = DatasetManager(**kwargs)
    # dataset = EncycDataset(dataset_name='hybridqa')
