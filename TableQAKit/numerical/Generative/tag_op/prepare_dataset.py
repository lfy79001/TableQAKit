import os
import pickle
import argparse
from tag_op.data.pipe import BartTatQATrainPipe, BartTatQATestPipe

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='./dataset_tagop')
parser.add_argument("--output_dir", type=str, default="./tag_op/cache")
parser.add_argument("--passage_length_limit", type=int, default=463)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--bart_name", type=str, default="plm/bart-base")
parser.add_argument("--mode", type=str, default='train')
args = parser.parse_args()

if args.mode == 'test':
#     data_reader = TagTaTQATestReader(tokenizer, args.passage_length_limit, args.question_length_limit, sep=sep)
    pipe = BartTatQATestPipe(tokenizer=args.bart_name)
    data_mode = ["test"]
elif args.mode == 'dev':
    pipe = BartTatQATrainPipe(tokenizer=args.bart_name)
    data_mode = ["dev"]
else:
    pipe = BartTatQATrainPipe(tokenizer=args.bart_name)
    data_mode = ["train"]

data_format = "tatqa_dataset_{}.json"
cache_data_format = "tagop_roberta_cached_{}.pkl"

for dm in data_mode:
    dpath = os.path.join(args.input_path, data_format.format(dm))
    output_path = os.path.join("tag_op/cache/", cache_data_format.format(dm))
    pipe.process_from_file(dpath, output_path, dm)
#     pipe.process(output_path)
