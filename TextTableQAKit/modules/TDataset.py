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
        
        
        return self.data[index]
        
        

class SpreadsheetDataset(Dataset):
    def __init__(self):
        pass
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return feature, label
    

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
                train_dataset = EncycDataset(dataset_data['train'], **kwargs)
                dev_dataset = EncycDataset(dataset_data['dev'], **kwargs)
                test_dataset = EncycDataset(dataset_data['test'], **kwargs)
                print(f"train_dataset: {len(train_dataset)}")
                print(f"dev_dataset: {len(dev_dataset)}")
                print(f"test_dataset: {len(test_dataset)}")
                pass
            elif kwargs['table_type'] == 'spreadsheet':
                pass
            elif kwargs['table_type'] == 'structured':
                pass
    def train():
        
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
    logger = create_logger()

    
    
    datasetManager = DatasetManager(**kwargs)
    # dataset = EncycDataset(dataset_name='hybridqa')
