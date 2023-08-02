"""
common classes for tools4wiki
"""
from secrets import choice
from typing import List, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
from utils.tapex_wikisql_utils import _TYPE_CONVERTER
from transformers import Seq2SeqTrainingArguments
import torch

DEFAULT_MODEL_ARGUMENTS = {
    'model_name_or_path': 'Yale-LILY/reastap-large',
}

DEFAULT_TRAINING_ARGUMENTS = {
    "do_train": True,
    "do_eval" : True,
    "output_dir": 'checkpoints/exp',
    "per_device_train_batch_size":16,
    "gradient_accumulation_steps":2,
    "per_device_eval_batch_size":8,
    "report_to": 'wandb',
    "num_train_epochs":20.0,
    "warmup_ratio":0.1,
    "learning_rate":3e-5,
    "fp16": True,
    "logging_steps":10,
    "eval_steps":200,
    "save_steps":400,
    "evaluation_strategy": 'steps',
    "predict_with_generate": True,
    "weight_decay": 1e-2,
    "label_smoothing_factor": 0.1,
    "generation_max_length": 128,
    "save_total_limit": 5,
    "run_name" : 'default',
    'max_source_length': 1024, 
    'max_target_length': 128, 
    "preprocessing_num_workers":16,
    'num_beams': 5, 
}

DEFAULT_EVAL_ARGUMENTS = {
    "do_eval" : True,
    "overwrite_output_dir": True,
    "max_source_length": 1024,
    "max_target_length": 128,
    "output_dir": 'outputs/eval',
    "per_device_eval_batch_size": 8,
    "predict_with_generate": True,
    "generation_max_length": 128,
    "num_beams": 5,
    "inference_set": 'validation'
}

DEFAULT_PREDICT_ARGUMENTS = {
    "do_predict" : True,
    "overwrite_output_dir": True,
    "max_source_length": 1024,
    "max_target_length": 128,
    "output_dir": 'outputs/predict',
    "per_device_eval_batch_size": 8,
    "predict_with_generate": True,
    "generation_max_length": 128,
    "num_beams": 5,
    "inference_set": 'test'
}

"""
From ReasTAP
"""

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pretrained tokenizer name or path if not the same as model_name. "
                "By default we use BART-large tokenizer for TAPEX-large."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

@dataclass
class TTQA_TrainingArguments(Seq2SeqTrainingArguments):
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    inference_set: Optional[str] = field(
        default="validation",
        metadata={
            "help": "Use which set for inference."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    def __post_init__(self):
        super().__post_init__() # **important**
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def convert_table_types(_table):
    """Runs the type converter over the table cells."""
    ret_table = deepcopy(_table)
    types = ret_table["types"]
    ret_table["real_rows"] = ret_table["rows"]
    typed_rows = []
    for row in ret_table["rows"]:
        typed_row = []
        for column, cell_value in enumerate(row):
            typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
        typed_rows.append(typed_row)
    ret_table["rows"] = typed_rows
    return ret_table

def postprocess_text(preds, labels):
    """remove whitespace at the begin and end"""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer, ignore_pad_token_for_loss = True):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    delimiter = ", "

    # define example evaluation
    def evaluate_example(predict_str: str, ground_str: str):
        predict_spans = predict_str.split(delimiter)
        ground_spans = ground_str.split(delimiter)
        predict_values = defaultdict(lambda: 0)
        ground_values = defaultdict(lambda: 0)
        for span in predict_spans:
            try:
                predict_values[float(span)] += 1
            except ValueError:
                predict_values[span.strip()] += 1
        for span in ground_spans:
            try:
                ground_values[float(span)] += 1
            except ValueError:
                ground_values[span.strip()] += 1
        is_correct = predict_values == ground_values
        return is_correct

    def get_denotation_accuracy(predictions: List[str], references: List[str]):
        assert len(predictions) == len(references)
        correct_num = 0
        for predict_str, ground_str in zip(predictions, references):
            is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
            if is_correct:
                correct_num += 1
        return correct_num / len(predictions)

    accuracy = get_denotation_accuracy(decoded_preds, decoded_labels)
    result = {"denotation_accuracy": accuracy}
    return result

def compute_metrics_tapas(eval_preds, tokenizer):
    """
    compute metrics for tapas model
    """
    logits, labels = eval_preds
    if hasattr(eval_preds,'aggregation_label_ids'):
        # TODO: 增加对聚合操作的预测评估
        pass

    preds = (logits > 0).astype(int) 
    acc_result = ~np.any(~(preds == labels), axis=-1)
    acc = sum(acc_result)/len(acc_result)
    result = {"denotation_accuracy": acc}
    return result
