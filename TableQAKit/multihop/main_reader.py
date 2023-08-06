import argparse
from Read import GenerateTrainer, MRCTrainer


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

    if args.mode == 'generate':
        args.plm = '/home/lfy/S3HQA1/PTM/bart-large'
        training_config = {}
        training_config['train_bs'] = 4
        training_config['dev_bs'] = 6
        training_config['epoch_num'] = 10
        training_config['max_len'] = 1024
        training_config['lr'] = 1e-5
        training_config['eps'] = 1e-8
        mytrainer = GenerateTrainer(args)
        mytrainer.train(**training_config)
    elif args.mode == 'mrc':
        args.plm = '/home/lfy/S3HQA1/PTM/bert-base-uncased'
        training_config = {}
        training_config['train_bs'] = 2
        training_config['dev_bs'] = 2
        training_config['epoch_num'] = 10
        training_config['max_len'] = 512
        training_config['lr'] = 1e-5
        training_config['eps'] = 1e-8
        mytrainer = MRCTrainer(args)
        mytrainer.train(**training_config)
    
    