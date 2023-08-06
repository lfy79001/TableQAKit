import argparse, json
from Read import GenerateTrainer, MRCTrainer

def read_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/train.row.json')
    parser.add_argument('--dev_data_path', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/dev.row.json')
    parser.add_argument('--dev_reference', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/dev_reference.json')
    parser.add_argument('--WikiTables', type=str, default='/home/lfy/S3HQA1/Data/HybridQA/WikiTables-WithLinks')
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='./try')
    parser.add_argument('--mode', choices=['mrc', 'generate'])
    parser.add_argument('--plm', type=str, default='/home/lfy/S3HQA1/PTM/bart-large')

    args = parser.parse_args()
    training_config = read_config(args.config_file)

    if args.mode == 'generate':
        mytrainer = GenerateTrainer(args)
        mytrainer.train(**training_config)
    elif args.mode == 'mrc':
        mytrainer = MRCTrainer(args)
        mytrainer.train(**training_config)
    
    