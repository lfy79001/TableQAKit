import sys
# sys.path.append('../')
import os
# if 'p' in os.environ:
#     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')
from tag_op.data.pipe import BartTatQATrainPipe
from tag_op.tagop.bart_absa import BartSeq2SeqModel

from fastNLP import Trainer
from tag_op.tagop.tatqa_s2s_metric import Seq2SeqSpanMetric
from tag_op.tagop.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from tag_op.tagop.generator import SequenceGeneratorModel
import fitlog
import torch
import torch.nn as nn

import torch.distributed as dist
from fastNLP import DistTrainer, get_local_rank

import argparse
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# fitlog.debug()
fitlog.set_log_dir('logs')


import argparse

def set_optimizer(model, args):
    parameters = []
    params = {'lr':args.lr, 'weight_decay':1e-2}
    params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
    parameters.append(params)

    params = {'lr':args.lr, 'weight_decay':1e-2}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    params = {'lr':args.lr, 'weight_decay':0}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)
    optimizer = optim.AdamW(parameters)
    
    return optimizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--blr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--opinion_first', action='store_true', default=False)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument('--bart_name', default='plm/bart-base', type=str)
    parser.add_argument('--use_encoder_mlp', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--max_len_a', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default='checkpoint') 
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--b_weight_decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=345)
    parser.add_argument('--local_rank', type=int, default=0)
    args= parser.parse_args()
    setup_seed(args.seed)
    print('args', args)
    
    fitlog.set_log_dir('logs')
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
        args.decoder_type = None
    decoder_type = args.decoder_type
    bart_name = args.bart_name
    use_encoder_mlp = args.use_encoder_mlp
    save_model = args.save_model
    fitlog.add_hyper(args)
    
    dist.init_process_group('nccl')
    if get_local_rank() != 0:
        dist.barrier() 
    
    #######hyper
    #######hyper
    pipe = BartTatQATrainPipe(tokenizer=args.bart_name)
    train_data_bundle = pipe.process(f'tag_op/cache/tagop_roberta_cached_train.pkl', 'train')
    dev_data_bundle = pipe.process(f'tag_op/cache/tagop_roberta_cached_dev.pkl', 'dev')
    tokenizer, mapping2id = pipe.tokenizer, pipe.mapping2id

    max_len = args.max_length
    max_len_a = args.max_len_a

    print("The number of tokens in tokenizer ", len(tokenizer.decoder))

    bos_token_id = 0  #
    eos_token_id = 1  #
    label_ids = list(mapping2id.values())
    model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                         copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)

    vocab_size = len(tokenizer)
    print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                   eos_token_id=eos_token_id,
                                   max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                   repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                   restricter=None)
    optimizer = set_optimizer(model, args)

#     # import torch
    if torch.cuda.is_available():
    #     device = list([i for i in range(torch.cuda.device_count())])
        device = 'cuda'
    else:
        device = 'cpu'

    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
    callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
#     callbacks.append(FitlogCallback(dev_data_bundle))

    sampler = None
    sampler = BucketSampler(seq_len_field_name='src_seq_len')
    metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids))

    model_path = None
    if save_model:
        model_path = args.save_model_path
    
    if get_local_rank() == 0:
        dist.barrier() 
    
    trainer = DistTrainer(train_data=train_data_bundle, model=model, optimizer=optimizer,
                      loss=Seq2SeqLoss(), batch_size_per_gpu=batch_size, sampler=sampler, 
                      drop_last=False, update_every=1,
                      num_workers=2, n_epochs=n_epochs, print_every=1,
                      dev_data=dev_data_bundle, metrics=metric, 
                      validate_every=-1, save_path=model_path, use_tqdm=True,
                      callbacks_all=callbacks, check_code_level=0, dev_batch_size=batch_size)

    trainer.train(load_best_model=False)


if __name__=="__main__":
    main()