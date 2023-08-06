import json
import argparse

from Retrieval import RetrievalTrainer

def read_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config



if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/train_r_col.json')
    parser.add_argument('--dev_data_path', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/dev_r_col.json')
    parser.add_argument('--WikiTables', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/WikiTables-WithLinks')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./try')
    parser.add_argument('--plm', type=str, default='/home/lfy/S3HQA1/PTM/bert-base-uncased')
    parser.add_argument('--mode', choices=['row', 'column'])
    parser.add_argument('--config_file', type=str, default='config_retriever.json')
    
    args = parser.parse_args()
    
    training_config = read_config(args.config_file)
    
    mytrainer = RetrievalTrainer(args)
    mytrainer.train(**training_config)
    
