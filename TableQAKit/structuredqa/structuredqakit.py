"""
Use this tool for different models and datasets to solve table related tasks.
"""
import pandas as pd
import logging
import os
import torch
import sys
from functools import partial
from typing import List, Optional
from abc import ABC, abstractmethod
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import json
from datasets import load_dataset
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TapexTokenizer,
    AutoTokenizer,
    TapasTokenizer,
    TapasForQuestionAnswering,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from utils.dataset import BaseStructuredQADataset
from utils.common import *

check_min_version("4.17.0.dev0")

logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"] = "reastap_march" # .

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class ToolBaseForStructuredQA(ABC):

    @abstractmethod
    def __init__(self, model_name_or_path, **kwargs) -> None:
        r"""
        对模型进行初始化, 加载模型权重, 
        固定 model 和 tokenizer

        Params:
            model_name_or_path (`str` or `os.PathLike`):
                Path or name to your model. Default: `Yale-LILY/reastap-large`
            tokenizer_name (`str` , *optional*):
                Name of tokenizer to be loaded. Default to model_name_or_path.
        """
        args_dict = DEFAULT_MODEL_ARGUMENTS.copy()
        args_dict['model_name_or_path'] = model_name_or_path
        args_dict.update(kwargs)
        parser = HfArgumentParser(ModelArguments)
        model_args,  = parser.parse_dict(args_dict)
        self.model_args = model_args
        self.dataset = None
        self.training_args = None
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        # Load pretrained model and tokenizer
        self._load_model(model_args)

    @abstractmethod
    def _load_model(self, model_args):
        print(f'Load Model: {model_args.model_name_or_path}')

    @abstractmethod
    def _init_running_cfg(self, default_cfg, **kwargs):
        pass
    
    # def run_wikisql(self):
    #     # TODO: 
    #     pass

    # def run_wtq(self):
    #     # TODO:
    #     pass

    # def run(self):
    #     # TODO:
    #     pass

    def train(self, 
              dataset: Optional[BaseStructuredQADataset] = None, 
              **kwargs):

        training_args = self._init_running_cfg(DEFAULT_TRAINING_ARGUMENTS,**kwargs)
        if dataset == None:
            raise ValueError("requires a dataset, like wikisql, wtq, e.g.")
        if dataset.train_dataset == None:
            raise ValueError("requires a train dataset")
        if dataset.eval_dataset == None:
            raise ValueError("requires a validation dataset")

        preprocess_fn = partial(self.preprocess_tableqa_function, sample_fn = dataset.get_example)
        train_dataset = dataset.train_dataset
        column_names = train_dataset.column_names

        if training_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(training_args.max_train_samples))
        preprocess_fn_training = partial(preprocess_fn, is_training=True)
        train_dataset = train_dataset.map(
            preprocess_fn_training,
            batched=True,
            num_proc=training_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not training_args.overwrite_cache,
        )
        eval_dataset = dataset.eval_dataset
        column_names = eval_dataset.column_names

        if training_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(training_args.max_eval_samples))
        preprocess_fn_eval = preprocess_fn
        eval_dataset = eval_dataset.map(
            preprocess_fn_eval,
            batched=True,
            num_proc=training_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not training_args.overwrite_cache,
        )
        self.last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            self.last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if self.last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif self.last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset= eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics
        
    def eval(self, 
              dataset: Optional[BaseStructuredQADataset] = None, 
              **kwargs):
        training_args = self._init_running_cfg(DEFAULT_EVAL_ARGUMENTS,**kwargs)
        if dataset == None:
            raise ValueError("requires a dataset, like wikisql, wtq, e.g.")
        
        preprocess_fn = partial(self.preprocess_tableqa_function, sample_fn = dataset.get_example)
        eval_dataset = dataset.eval_dataset
        column_names = eval_dataset.column_names
        
        if training_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(training_args.max_eval_samples))

        preprocess_fn_eval = preprocess_fn


        ########################
        ## inject: test tapas tokenizer
        # BEGIN = 4872
        # BATCH_SIZE = 12
        # for i in range(BEGIN,BEGIN+BATCH_SIZE): ## [1480, 4881] 挂了
        #     print(i)
        #     row_data = eval_dataset[i]

        #     sample_input = dict()

        #     for key, value in row_data.items():
        #         # print(key)
        #         sample_input[key] = [value]
            
        #     input = preprocess_fn_eval(examples=sample_input)

        #     model = self.model
        #     model_input = input
        #     # model_input = dict()

        #     # for key, value in input.items():
        #     #     import pdb; pdb.set_trace()
        #     #     if key in ['input_ids', 'labels', 'token_type_ids', 'attention_mask']:
        #     #         model_input[key] = torch.tensor(value, dtype=torch.int64)
        #     #     else:
        #     #         model_input[key] = torch.tensor(value)
        #     for key, value in input.items():
        #         if value.max() == 151:
        #             import pdb; pdb.set_trace()
            
        #     if len(input)==0:
        #         continue
        #     try:
        #         outputs = model(**model_input)
        #     except Exception as e:
        #         print(e)
        #         import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        #########################

        eval_dataset = eval_dataset.map(
            preprocess_fn_eval,
            batched=True,
            num_proc=training_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not training_args.overwrite_cache,
        )

        ########################
        ## inject: test tapas model
        # for i in range(980,983):
        #     input = eval_dataset[i]
        #     model = self.model
        #     model_input = dict()
        #     for key, value in input.items():
        #         if key in ['input_ids', 'labels', 'token_type_ids', 'attention_mask']:
        #             model_input[key] = torch.tensor([value], dtype=torch.int64)
        #         else:
        #             model_input[key] = torch.tensor([value])
        #     import pdb; pdb.set_trace()
        #     outputs = model(**model_input)
        #     loss = outputs.loss
        #     pdb.set_trace()
        #########################

        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset= None,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        # Evaluation
        # results = {}
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=training_args.val_max_target_length, num_beams=training_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = training_args.max_eval_samples if training_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)

        return metrics

    def predict(self, dataset=None, **kwargs):
        training_args = self._init_running_cfg(DEFAULT_PREDICT_ARGUMENTS, **kwargs)
        max_target_length = training_args.val_max_target_length

        if dataset == None:
            raise ValueError("requires a dataset, like wikisql, wtq, e.g.")

        preprocess_fn = partial(self.preprocess_tableqa_function, sample_fn = dataset.get_example)
        test_dataset = dataset.test_dataset
        raw_test_dataset = test_dataset
        column_names = test_dataset.column_names
        if training_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(training_args.max_predict_samples))
        preprocess_fn_test = preprocess_fn
        test_dataset = test_dataset.map(
            preprocess_fn_test,
            batched=True,
            num_proc=training_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not training_args.overwrite_cache,
        )
        
        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset= None,
            eval_dataset= None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            max_length=training_args.val_max_target_length,
            num_beams=training_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            training_args.max_predict_samples if training_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        
        predictions = self.tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]

        outputs = []

        # NOTE: 这里的答案应该只针对 wikisql 和 wtq 数据集, 别的应该没有
        # for i, pred in enumerate(predictions):
        #     example = raw_test_dataset[i]
        #     if self.seek_sql:
        #         table = convert_table_types(example["table"])
        #         answer = retrieve_wikisql_query_answer_tapas(table, example["sql"])

        #         answer = ", ".join([i.lower() for i in answer])
        #         example_id = str(i)
        #     else:
        #         answer = ", ".join([i.lower() for i in example["answers"]])
        #         example_id = example["id"]

        #     outputs.append({"id": example_id, "prediction": pred, "answers": answer})

        for i, pred in enumerate(predictions):
            example = raw_test_dataset[i]
            if 'id' in example.keys():
                outputs.append({"id": example["id"], "prediction": pred})
            else :
                outputs.append({"id": str(i), "prediction": pred}) 

        model_name = self.model_args.model_name_or_path.split("/")[-1].split("-")[0]
        if training_args.output_dir:
            output_prediction_file = os.path.join(training_args.output_dir, f"{model_name}_{dataset.name}_{training_args.inference_set}_predictions.json")
            json.dump(outputs, open(output_prediction_file, "w"), indent=4)
        
        return outputs


    @abstractmethod
    def preprocess_tableqa_function(self, examples, sample_fn = None, is_training=False):
        pass
    

class BertForStructuredQA(ToolBaseForStructuredQA):
    """
    Bert-based TableQA Tools (TAPAS)
    """
    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)


    def _load_model(self, model_args):
        super()._load_model(model_args)
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # NOTE: we do not use `aggregation_labels` now
        config.num_aggregation_labels = 0

        self.tokenizer = TapasTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )

        self.model = TapasForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    def _init_running_cfg(self, default_cfg, **kwargs):
        """
        initialize running config and data collator.
        special for bert-based model.
        """
        args_dict = default_cfg.copy()
        args_dict["max_source_length"] = 512
        args_dict["max_target_length"] = None
        args_dict["predict_with_generate"] = False
        args_dict["pad_to_max_length"]= True
        args_dict["val_max_target_length"] = None
        args_dict["label_names"] = ['labels'] # NOTE: default to use only `labels` as labels
        args_dict.update(kwargs)

        parser = HfArgumentParser(TTQA_TrainingArguments)

        training_args, = parser.parse_dict(args_dict)
        self.training_args = training_args
        
        logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
        logger.info(f"Training/evaluation parameters {training_args}")

        # Set seed before initializing model.
        set_seed(training_args.seed)
        # Temporarily set max_target_length for training.
        self.padding = "max_length" if training_args.pad_to_max_length else False

        if training_args.label_smoothing_factor > 0 and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self.model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )
        
        # Data collator
        self.label_pad_token_id = -100 if training_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        compute_metrics_fn = args_dict['compute_metrics'] if 'compute_metrics' in args_dict else partial(compute_metrics_tapas, tokenizer = self.tokenizer)

        self.compute_metrics = compute_metrics_fn
        return training_args

    def preprocess_tableqa_function(self, examples, sample_fn = None, is_training=False):
        """
        The is_training FLAG is used to identify if we could use the supervision
        to truncate the table content if it is required.

        seek_sql FLAG is used to identify whether to retrieve the WikiSQL answer for each question
        """
        if sample_fn == None:
            raise ValueError("should have a sample_fn")

        questions, tables, answers, answer_coordinates = sample_fn(examples)
        # IMPORTANT: we cannot pass by answers during evaluation, answers passed during training are used to
        # truncate large tables in the train set!

        # NOTE: tapas tokenizer only support tokenizing one table one time
        model_inputs = dict()

        
        # if is_training: # NOTE: it seems useless in tapas
        if answers is not None and len(answers) > 0:
            for table, query, answer,answer_coordinates in zip(tables, questions, answers, answer_coordinates):
                tuple_coordinates_list = [tuple(item) for item in answer_coordinates]
                try:
                    tokenize_result = self.tokenizer(
                        table=table.astype(str),
                        queries=query,
                        answer_text=answer,
                        answer_coordinates = tuple_coordinates_list,
                        max_length=self.training_args.max_source_length,
                        padding=self.padding,
                        truncation=True,
                        return_tensors="pt",
                    )
                except:
                    print("WARNING: One table is too large to tokenize, remove it.")
                    continue

                if tokenize_result['token_type_ids'][:,:,4:6].max() > 127: # TODO: fix tokenizer
                    print("WARNING: There is a dirty data, remove it.")
                    continue

                for (key ,value) in tokenize_result.items():
                    if key not in model_inputs.keys():
                        model_inputs[key] = value
                    else:
                        model_inputs[key] = torch.cat((model_inputs[key],value),dim=0)
        else:
            for table, query in zip(tables, questions):
                tokenize_result = self.tokenizer(
                    table=table.astype(str),
                    queries=query,
                    max_length=self.training_args.max_source_length,
                    padding=self.padding,
                    truncation=True,
                    return_tensors="pt",
                )
                for (key ,value) in tokenize_result.items():
                    if key not in model_inputs.keys():
                        model_inputs[key] = value
                    else:
                        model_inputs[key] = torch.cat((model_inputs[key],value),dim=0)

        return model_inputs



class BartForStructuredQA(ToolBaseForStructuredQA):
    """
    Bart-based TextTableQA Tools
    """

    def __init__(self, model_name_or_path, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)
    
    def _load_model(self, model_args):
        super()._load_model(model_args)
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        config.max_length = 1024
        config.early_stopping = False

        # load tapex tokenizer
        self.tokenizer = TapexTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )

        # load Bart based Tapex model (default tapex-large)
        self.model = BartForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    def _init_running_cfg(self, default_cfg, **kwargs):
        args_dict = default_cfg.copy()
        args_dict.update(kwargs)
        parser = HfArgumentParser(TTQA_TrainingArguments)
        training_args, = parser.parse_dict(args_dict)
        self.training_args = training_args
        
        logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
        logger.info(f"Training/evaluation parameters {training_args}")

        # Set seed before initializing model.
        set_seed(training_args.seed)
        # Temporarily set max_target_length for training.
        self.max_target_length = training_args.max_target_length
        self.padding = "max_length" if training_args.pad_to_max_length else False

        if training_args.label_smoothing_factor > 0 and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self.model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )
        
        # Data collator
        self.label_pad_token_id = -100 if training_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        compute_metrics_fn = args_dict['compute_metrics'] if 'compute_metrics' in args_dict else compute_metrics

        self.compute_metrics = partial(compute_metrics_fn, tokenizer = self.tokenizer) if training_args.predict_with_generate else None

        return training_args


    def preprocess_tableqa_function(self, examples, sample_fn = None, is_training=False):
        """
        The is_training FLAG is used to identify if we could use the supervision
        to truncate the table content if it is required.

        seek_sql FLAG is used to identify whether to retrieve the WikiSQL answer for each question
        """
        if sample_fn == None:
            raise ValueError("should have a sample_fn")
        
        questions, tables, answers = sample_fn(examples)[:3]

        # IMPORTANT: we cannot pass by answers during evaluation, answers passed during training are used to
        # truncate large tables in the train set!
        if is_training:
            model_inputs = self.tokenizer(
                table=tables,
                query=questions,
                answer=answers,
                max_length=self.training_args.max_source_length,
                padding=self.padding,
                truncation=True,
            )
        else:
            model_inputs = self.tokenizer(
                table=tables, query=questions, max_length=self.training_args.max_source_length, padding=self.padding, truncation=True
            )
            
        labels = self.tokenizer(
            answer=[", ".join(answer) for answer in answers],
            max_length=self.max_target_length,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.training_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

if __name__ == '__main__':
    tool = BertForStructuredQA(model_name_or_path = '/home/lfy/PTM/tapex-large-finetuned-sqa')