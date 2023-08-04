import json
import argparse

from Retrieval import RetrievalTrainer

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



if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/train.json')
    parser.add_argument('--dev_data_path', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/dev.json')
    parser.add_argument('--WikiTables', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/WikiTables-WithLinks')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./try')
    parser.add_argument('--plm', type=str, default='/home/lfy/S3HQA1/PTM/bert-base-uncased')
    
    args = parser.parse_args()
    
    training_config = {}
    training_config['train_bs'] = 1
    training_config['dev_bs'] = 1
    training_config['epoch_num'] = 5
    training_config['max_len'] = 512
    training_config['lr'] = 7e-6
    training_config['eps'] = 1e-8
    mytrainer = RetrievalTrainer(args)
    mytrainer.train(**training_config)
    
