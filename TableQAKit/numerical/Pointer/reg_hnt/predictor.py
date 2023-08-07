import io, requests, zipfile
import os
import json
import argparse
from datetime import datetime
from reg_hnt import options
import torch
import torch.nn as nn
from pprint import pprint
from reg_hnt.data.data_util import OPERATOR_CLASSES_
from reg_hnt.reghnt.util import create_logger, set_environment
from reg_hnt.data.tatqa_batch_gen import TaTQABatchGen
from reg_hnt.data.tatqa_test_batch_gen import TaTQATestBatchGen
from transformers import RobertaModel, BertModel, ElectraModel, AutoModel
from reg_hnt.reghnt.modeling_reghnt import RegHNTModel
from reg_hnt.reghnt.model import RegHNTFineTuningModel, RegHNTPredictModel
from pathlib import Path
from reg_hnt.reghnt.Vocabulary import *
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): # add this line
            return obj.tolist() # add this line
        return json.JSONEncoder.default(self, obj) 



parser = argparse.ArgumentParser("RegHNT predicting task.")
options.add_data_args(parser)
options.add_bert_args(parser)
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--model_path", type=str, default="checkpoint")
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--encoder", type=str, default='roberta_large')
parser.add_argument("--test_data_dir", type=str, default="reg_hnt/cache/")
parser.add_argument("--mode", type=str, default='test')
parser.add_argument("--step", type=str, default='op')

args = parser.parse_args()
if args.ablation_mode != 0:
    args.model_path = args.model_path + "_{}_{}".format(args.op_mode, args.ablation_mode)
if args.ablation_mode != 0:
    args.data_dir = args.data_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

args.cuda = args.gpu_num > 0

logger = create_logger("RegHNT Predictor", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.cuda)

def main():
    if args.mode == 'dev':
        dev_itr = TaTQABatchGen(args, data_mode='dev', encoder=args.encoder)
    elif args.mode == 'test':
        dev_itr = TaTQATestBatchGen(args, data_mode='test', encoder=args.encoder)

    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained(args.bert_model)
    elif args.encoder == 'roberta_base':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'roberta_large':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
        
    if args.ablation_mode == 0:
        operators = OPERATOR_CLASSES_

    if args.ablation_mode == 0:
        arithmetic_op_index = [3, 4, 6, 7, 8, 9]

    network = RegHNTModel(
        encoder = bert_model,
        config = bert_model.config,
        bsz = args.eval_batch_size,
        operator_classes = len(OPERATOR_CLASSES),
        scale_classes = len(SCALE),
        operator_criterion = nn.CrossEntropyLoss(),
        scale_criterion = nn.CrossEntropyLoss(),
        arithmetic_op_index = arithmetic_op_index,
        op_mode = args.op_mode,
        ablation_mode = args.ablation_mode,
    )

    network.load_state_dict(torch.load(os.path.join(args.model_path,"checkpoint_best.pt")))
    model = RegHNTPredictModel(args, network)
    
    logger.info("Below are the result on Dev set...")
    model.reset()
    model.avg_reset()

    if args.mode == 'dev':
        pred_json = model.predict(dev_itr)
        model.get_metrics()
        json.dump(pred_json,open(os.path.join(args.save_dir, 'pred_result_on_dev.json'), 'w'),ensure_ascii=False,cls=NpEncoder)
    elif args.mode == 'test':
        pred_json = model.predict2(dev_itr)
        json.dump(pred_json,open(os.path.join(args.save_dir, 'pred_result_on_test.json'), 'w'),ensure_ascii=False,cls=NpEncoder)



if __name__ == "__main__":
    main()