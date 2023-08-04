from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json


class RetrievalDataset(Dataset):
    def __init__(self, args, data):
        super(RetrievalDataset, self).__init__()
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
        self.tokenizer = tokenizer
        self.total_data = []
        for item in tqdm(data): # 看字典的元素，xxx.keys()
            
            if item['labels'].count(1) == 0:
                continue
            
            question = item['question']
            path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
            table_id = item['table_id']
            
            
            with open(f'{path}/tables_tok/{table_id}.json', 'r') as f:
                table = json.load(f) 
            
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            
            item_data = {}
            input_ids = [] 
            question_ids = tokenizer.encode(question)
            template = '{} is {}. '
            headers =  [aa[0] for aa in table['header']]
            
            for i, row in enumerate(table['data']):
                 
                links = []
                row_ids = []
                for j, cell in enumerate(row):
                    
                    cell_string = template.format(headers[j], cell[0]) #cell[1]代表超链接
                    cell_tokens = tokenizer.tokenize(cell_string)
                    cell_ids = tokenizer.convert_tokens_to_ids(cell_tokens)
                    row_ids.extend(cell_ids)
                    row_ids += [tokenizer.sep_token_id] #加分隔符sep
                    
                    links.extend(cell[1])
                
                passage_ids = []
                for link in links:
                    
                    link_tokens = tokenizer.tokenize(requested_document[link]) #超链接
                    link_ids = tokenizer.convert_tokens_to_ids(link_tokens)
                    passage_ids.extend(link_ids)
                    passage_ids += [tokenizer.sep_token_id]
                
                
                row_ids = question_ids + row_ids + passage_ids
                input_ids.append(row_ids)
            
            item_data['input_ids'] = input_ids
            item_data['labels'] = item['labels']
            item_data['metadata'] = item #原数据
            self.total_data.append(item_data) # 标签加input
            
    def __len__(self):
        return len(self.total_data)
    def __getitem__(self, index):
        data = self.total_data[index]
        return data
    
    



# def hybridqa_content(data):
#     path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
#     table_id = data['table_id']
#     with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
#         table = json.load(f)  
#     with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
#         requested_document = json.load(f)
#     content = []
#     for i, row in enumerate(table['data']):
#         # read the cell of each row, put the in a list. For special HybridQA Dataset, read the cell links.
#         cells_i = [item[0] for item in row]
#         links_iii = [item[1] for item in row]
#         links_ii = [item for sublist in links_iii for item in sublist]
#         links_i = [requested_document[link] for link in links_ii]
#         content.append((cells_i, links_i))
#     return content

# def hybridqa_header(data):
#     path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
#     table_id = data['table_id']
#     with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
#         table = json.load(f)  
#     with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
#         requested_document = json.load(f)
#     header = [_[0] for _ in table['header']]
#     return header   

# def hybridqa_label(data):
#     return data['labels']