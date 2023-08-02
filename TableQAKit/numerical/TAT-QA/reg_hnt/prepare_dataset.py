import os
import pickle
import argparse
from reg_hnt.data.tatqa_dataset import TaTQAReader
from reg_hnt.data.tatqa_dataset_test import TaTQATestReader
from transformers import RobertaTokenizer
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='./dataset_reghnt')
parser.add_argument("--output_dir", type=str, default="./reg_hnt/cache")
parser.add_argument("--passage_length_limit", type=int, default=512)
parser.add_argument("--question_length_limit", type=int, default=60)
parser.add_argument("--encoder", type=str, default="roberta_large")
parser.add_argument("--mode", type=str, default='train')

args = parser.parse_args()

if args.encoder == 'roberta_base':
    tokenizer = RobertaTokenizer.from_pretrained(args.input_path + "/roberta.base")
    sep = '<s>'
elif args.encoder == 'roberta_large':
    tokenizer = RobertaTokenizer.from_pretrained(args.input_path + "/roberta.large")
    sep = '<s>'
elif args.encoder == 'bert':
    tokenizer = BertTokenizer.from_pretrained(args.input_path + "/bert.large")
    sep = '[SEP]'


if args.mode == 'train':
    data_reader =  TaTQAReader(tokenizer, args.passage_length_limit, args.question_length_limit, mode='train', sep=sep)
    data_mode = {"train"}
elif args.mode == 'dev':
    data_reader =  TaTQAReader(tokenizer, args.passage_length_limit, args.question_length_limit, mode='dev', sep=sep)
    data_mode = {"dev"}
elif args.mode == 'test':
    data_reader = TaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = {"test"}
elif args.mode == 'test1':
    data_reader = TaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    data_mode = {"test1"}

data_format = "tatqa_dataset_{}.json"
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')

for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    data = data_reader._read(dpath)
    print(data_reader.skip_count)
    data_reader.skip_count = 0
    print("Save data to {}.".format(os.path.join(args.output_dir, f"reghnt_{args.encoder}_cached_{dm}.pkl")))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, f"reghnt_{args.encoder}_cached_{dm}.pkl"), "wb") as f:
        pickle.dump(data, f)
