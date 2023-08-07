import sys
sys.path.append('/export/users/zhouyongwei/UniRPG_full')
import os
# if 'p' in os.environ:
#     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')
from tag_op.data.pipe import BartTatQATrainPipe, BartTatQATestPipe
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
from fastNLP import Tester
import pickle
import json

# fitlog.debug()
fitlog.set_log_dir('logs')

import torch
import time
from tqdm import tqdm
import argparse
import torch.nn as nn
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.utils import _move_model_to_device
from fastNLP.core._logger import logger
from fastNLP import DataSet
from fastNLP.core.batch import BatchIter, DataSetIter
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.utils import _CheckError
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_loss_evaluate
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.utils import _get_func_signature
from fastNLP.core.utils import _get_model_device
from fastNLP.core.utils import _move_model_to_device
from fastNLP.core.utils import _build_fp16_env
from fastNLP.core.utils import _can_use_fp16
from fastNLP.core._parallel_utils import _data_parallel_wrapper
from fastNLP.core._parallel_utils import _model_contains_inner_module
from functools import partial
from fastNLP.core._logger import logger
from fastNLP.core.sampler import Sampler



class MyTester(Tester):
    def __init__(self, data, model, metrics, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True, fp16=False, **kwargs):
        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")
        
        self.metrics = _prepare_metrics(metrics)
        
        self.data = data
        self._model = _move_model_to_device(model, device=device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.logger = logger
        self.pin_memory = kwargs.get('pin_memory', True)

        if isinstance(data, DataSet):
            sampler = kwargs.get('sampler', None)
            if sampler is None:
                sampler = SequentialSampler()
            elif not isinstance(sampler, (Sampler, torch.utils.data.Sampler)):
                raise ValueError(f"The type of sampler should be fastNLP.BaseSampler or pytorch's Sampler, got {type(sampler)}")
            if hasattr(sampler, 'set_batch_size'):
                sampler.set_batch_size(batch_size)
            self.data_iterator = DataSetIter(dataset=data, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers,
                                             pin_memory=self.pin_memory)
        elif isinstance(data, BatchIter):
            self.data_iterator = data
        else:
            raise TypeError("data type {} not support".format(type(data)))

        # check predict
        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                    self._model.device_ids,
                                                                    self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict  
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict  
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
        else:
            if _model_contains_inner_module(self._model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward

        if fp16:
            _can_use_fp16(model=model, device=device, func=self._predict_func)
        self.auto_cast, _grad_scaler = _build_fp16_env(not fp16)
        

    def test(self):
        # turn on the testing mode; clean up the history
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        predict_result = {}
        predict_scale_result = {}
        
        predict_scale_result = {}
        predict_scores = {}
        
        
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")
                    start_time = time.time()
                    for batch_x, batch_y in data_iterator:
                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device,
                                                   non_blocking=self.pin_memory)
                        with self.auto_cast():
                            pred_dict = self._data_forward(self._predict_func, batch_x)
                            if not isinstance(pred_dict, dict):
                                raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                                f"must be `dict`, got {type(pred_dict)}.")
                            for metric in self.metrics:
                                metric(pred_dict, batch_y)
                            predict_result_batch = pred_dict["pred"][0].detach().cpu().numpy().tolist()
                            scale_result_batch = pred_dict["pred"][1].detach().cpu().numpy().tolist()
                            predict_scores_batch = pred_dict["pred"][2].detach().cpu().numpy().tolist()
                            
                            
                            predict_result.update(dict(zip(batch_x['question_id'], predict_result_batch)))
                            predict_scale_result.update(dict(zip(batch_x['question_id'], scale_result_batch)))
                            predict_scores.update(dict(zip(batch_x['question_id'], predict_scores_batch)))
                            
                            
                            
                            
                            
                        if self.use_tqdm:
                            pbar.update()

                    for metric in self.metrics:
                        eval_result = metric.get_metric()
                        if not isinstance(eval_result, dict):
                            raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                            f"`dict`, got {type(eval_result)}")
                        metric_name = metric.get_metric_name()
                        eval_results[metric_name] = eval_result
                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        finally:
            self._mode(network, is_test=False)
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        
        import pdb
        pdb.set_trace()
        return eval_results, predict_result, predict_scale_result, predict_scores
    
    
    
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
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--opinion_first', action='store_true', default=False)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument('--bart_name', default='plm/bart-base', type=str)
    parser.add_argument('--use_encoder_mlp', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--model_path', type=str, default= "checkpoint/best_SequenceGeneratorModel_em_2021-12-24-18-21-23-663758")
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--max_len_a', type=int, default=0)
    parser.add_argument('--add_structure', type=int, default=0)
    
    args= parser.parse_args()

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

    #######hyper
    #######hyper
    pipe = BartTatQATestPipe(tokenizer=args.bart_name)
    test_data_bundle = pipe.process(f'tag_op/cache/tagop_roberta_cached_test.pkl', 'test')
    tokenizer, mapping2id = pipe.tokenizer, pipe.mapping2id

    max_len = args.max_length
    max_len_a = args.max_len_a
    print("The number of tokens in tokenizer ", len(tokenizer.decoder))

    bos_token_id = 0  #
    eos_token_id = 1  #
    label_ids = list(mapping2id.values())
    model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                         copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False, add_structure=args.add_structure)

    vocab_size = len(tokenizer)
    print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                   eos_token_id=eos_token_id,
                                   max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                   repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                   restricter=None)
    optimizer = set_optimizer(model, args)

    # import torch
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
    callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))

    sampler = None
    sampler = BucketSampler(seq_len_field_name='src_seq_len')

    model_path = args.model_path
    model = torch.load(model_path)
    tester = MyTester(data=test_data_bundle, model=model, batch_size=batch_size,
                      num_workers=2, metrics=None, use_tqdm=True, device=device,
                      test_sampler=SortedSampler('src_seq_len'))
    eval_result, predict_result, predict_scale_result, predict_scores = tester.test()
    predict_grammar_result = get_grammar_from_id(pipe.mapping2targetid, predict_result)
    result = {}
    
    for key in predict_grammar_result.keys():
        result.update({ key:{"predict_grammar":predict_grammar_result[key], "pred_scale": predict_scale_result[key], "predict_score":predict_scores[key] }} )
       
    with open(os.path.join(os.path.dirname(args.model_path),'predict_grammar_test.json'), 'w') as f:
        json.dump(result, f, indent=4)
        print("finish write predict result")
    f.close()
    
    
    
    
def get_grammar_from_id(mapping2targetid, predict_result):
    targetid2grammar = dict(zip(mapping2targetid.values(), mapping2targetid.keys()))
    start_end_id = {
        0:'<s>',
        1:'</s>'
    }
    target_shift = len(targetid2grammar) + 2
    
    predict_grammar_result = {}
    import pdb
#     pdb.set_trace()
    for ins_idx, ins_predict in predict_result.items():

        ins_predict_grammar = []
        for item in ins_predict:
            if item in start_end_id.keys():
                ins_predict_grammar.append(start_end_id[item])
            elif 2<=item<=target_shift-1:
                ins_predict_grammar.append(targetid2grammar[item-2])
            else:
                ins_predict_grammar.append(str(item-target_shift))
#         pdb.set_trace()
        ins_predict_grammar = ' '.join(ins_predict_grammar)
        predict_grammar_result.update({ins_idx:ins_predict_grammar})
    return  predict_grammar_result
    
    
    
    
    
    
    
if __name__=="__main__":
    main()
