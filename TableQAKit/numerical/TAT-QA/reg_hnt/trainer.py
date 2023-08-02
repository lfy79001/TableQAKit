import io, requests, zipfile
import os
import json
import argparse
from datetime import datetime
from reg_hnt import options
import torch.nn as nn
from pprint import pprint
from reg_hnt.data.data_util import OPERATOR_CLASSES_
from reg_hnt.reghnt.util import create_logger, set_environment
from reg_hnt.reghnt.Vocabulary import *
from reg_hnt.data.tatqa_batch_gen import TaTQABatchGen
from transformers import RobertaModel, BertModel, ElectraModel, AutoModel
from reg_hnt.reghnt.modeling_reghnt import RegHNTModel
from reg_hnt.reghnt.model import RegHNTFineTuningModel
from pathlib import Path
import torch
parser = argparse.ArgumentParser("RegHNT training task.")
options.add_data_args(parser)
options.add_train_args(parser)
options.add_bert_args(parser)
parser.add_argument("--encoder", type=str, default='roberta_large')
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--test_data_dir", type=str, default="./reg_hnt/cache")

args = parser.parse_args()
if args.ablation_mode != 0:
    args.save_dir = args.save_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)
    args.data_dir = args.data_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)

Path(args.save_dir).mkdir(parents=True, exist_ok=True)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps

logger = create_logger("Roberta Training", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)

def main():
    best_result = float("-inf")
    logger.info("Loading data...")

    train_itr = TaTQABatchGen(args, data_mode = "train", encoder=args.encoder)
    if args.ablation_mode != 3:
        dev_itr = TaTQABatchGen(args, data_mode="dev", encoder=args.encoder)
    else:
        dev_itr = TaTQABatchGen(args, data_mode="dev", encoder=args.encoder)

    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
    logger.info("Num update steps {}!".format(num_train_steps))

    logger.info(f"Build {args.encoder} model.")
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
        bsz = args.batch_size,
        operator_classes = len(OPERATOR_CLASSES),
        scale_classes = len(SCALE),
        operator_criterion = nn.CrossEntropyLoss(),
        scale_criterion = nn.CrossEntropyLoss(),
        arithmetic_op_index = arithmetic_op_index,
        op_mode = args.op_mode,
        ablation_mode = args.ablation_mode,
    )
    logger.info("Build optimizer etc...")

    model = RegHNTFineTuningModel(args, network, num_train_steps=num_train_steps)

    train_start = datetime.now()
    first = True
    for epoch in range(1, args.max_epoch + 1):
        model.reset()
        if not first:
            train_itr.reset()
        first = False
        logger.info('At epoch {}'.format(epoch))
        for step, batch in enumerate(train_itr):
            model.update(batch)
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                logger.info("Updates[{0:6}] train loss[{1:.5f}] remaining[{2}].\r\n".format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))
                model.avg_reset()
        model.get_metrics(logger)

        model.reset()
        model.avg_reset()
        model.predict(dev_itr)
        metrics = model.get_metrics(logger)
        model.avg_reset()
        if metrics["f1"] > best_result:
            save_prefix = os.path.join(args.save_dir, "checkpoint_best")
            model.save(save_prefix, epoch)
            best_result = metrics["f1"]
            logger.info("Best eval F1 {} at epoch {}.\r\n".format(best_result, epoch))

if __name__ == "__main__":
    main()