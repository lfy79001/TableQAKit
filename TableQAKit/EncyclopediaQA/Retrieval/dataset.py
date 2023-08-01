from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


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
    
    
