import torch
import torch.nn as nn
from torch.utils.data import Dataset

        
        
class EncycDataset(Dataset):
    def __init__(self, **kwargs):
        import pdb; pdb.set_trace()
        dataset_name = None
        if 'dataset_name' in kwargs.keys():
            dataset_name = kwargs['dataset_name']
        if dataset_name != None:
            
        
    
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
    
if __name__ == '__main__':
    dataset = EncycDataset(dataset_name='hybridqa')
