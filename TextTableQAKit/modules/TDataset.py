import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import *
from transformers import AutoTokenizer
from tqdm import tqdm


# question_id, question, answer,   
class EncycDataset(Dataset):
    def __init__(self, all_data, **kwargs):
        super(EncycDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['plm'])
        self.data = []
        for item in tqdm(all_data):
            import pdb; pdb.set_trace()
            data_i = {}
            data_i
            
            
        
        
    def __len__(self):
        pass
        
    def __getitem__(self, index):
        data = self.data[index]
        
        

class SpreadsheetDataset(Dataset):
    def __init__(self):
        pass
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return feature, label
    
    
    
class DatasetManager:
    def __init__(self, **kwargs):
        if kwargs['type'] == 'common': # 使用标准的数据集
            # get the path of the dataset
            dataset_path = dataset_download(kwargs['dataset_name'])
            # load data  -> dict {'train', 'dev', 'test'}
            dataset_data = load_data(dataset_path)
            if kwargs['table_type'] == 'encyc':
                train_dataset = EncycDataset(dataset_data['train'], **kwargs)
                dev_dataset = EncycDataset(dataset_data['dev'], kwargs)
                test_dataset = EncycDataset(dataset_data['test'], kwargs)
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
    kwargs['plm'] = 'bert-base-uncased'
    kwargs['dataset_config'] = {'id': 'question_id', 'question': 'question', 'answer': 'answer-text'}
    
    # need to write a function map the table to row
    
    
    def aa():
        print(11)
        
    aa()
    
    
    datasetManager = DatasetManager(**kwargs)
    # dataset = EncycDataset(dataset_name='hybridqa')
