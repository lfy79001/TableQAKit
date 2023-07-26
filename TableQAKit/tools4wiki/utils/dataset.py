import os
import torch
from torch.utils.data import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from transformers import HfArgumentParser
from datasets import load_dataset
from typing import Tuple, List, Optional
import pandas as pd
from utils.wikisql_utils import _TYPE_CONVERTER, retrieve_wikisql_query_answer_tapas
from utils.common import convert_table_types

INTRINSIC_DATASETS = ['wikisql','wikitablequestions']

def load_intrinsic_dataset(dataset_name = None, dataset_config_name = None, cache_dir = None):
    datasets = load_dataset(dataset_name, dataset_config_name, cache_dir= cache_dir)
    return datasets

def load_dataset_from_file(train_file = None, validation_file = None, test_file = None, cache_dir = None):
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]
    if validation_file is not None:
        data_files["validation"] = validation_file
        extension = validation_file.split(".")[-1]
    if test_file is not None:
        data_files["test"] = test_file
        extension = test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
    return datasets


# class BaseTableTextQADataset(Dataset):
class BaseTableTextQADataset:
    """
    BaseDataset for Table-Text Question Answering tasks
    """
    def __init__(self, cfg_file = None, **kwargs) -> None:
        super().__init__()
        self.datasets = {
            'train': None,
            'validation': None,
            'test': None
        }


    @property
    def train_dataset(self):
        return self.datasets['train']

    @property
    def eval_dataset(self):
        return self.datasets['validation']

    @property
    def test_dataset(self):
        return self.datasets['test']
    
    @staticmethod
    def get_example(examples, split) -> tuple:
        """
        a function to get (question, table, answer, ...)
        """
        raise NotImplementedError


class WikisqlDataset(BaseTableTextQADataset):

    def __init__(self, dataset_name='wikisql', dataset_config_name=None, cache_dir=None, **kwargs) -> None:
        super().__init__(**kwargs)
        # raw_datasets_split: DatasetDict = load_intrinsic_dataset(dataset_name, dataset_config_name, cache_dir)
        self.datasets = load_intrinsic_dataset(dataset_name, dataset_config_name, cache_dir)

    @staticmethod
    def get_example(examples) -> tuple:
        questions = [question.lower() for question in examples["question"]]
        example_tables = examples["table"]
        example_sqls = examples["sql"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        # answer set
        answers = []
        for example_sql, example_table in zip(example_sqls, example_tables):
            tapas_table = convert_table_types(example_table)
            answer_list: List[str] = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)
            # you can choose other delimiters to split each answer
            answers.append(answer_list)
        
        return (questions, tables, answers)

class WikitablequestionsDataset(BaseTableTextQADataset):

    def __init__(self, dataset_name='wikitablequestions', dataset_config_name=None, cache_dir=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.datasets = load_intrinsic_dataset(dataset_name, dataset_config_name, cache_dir)

    @staticmethod
    def get_example(examples) -> tuple:
        questions = [question.lower() for question in examples["question"]]
        example_tables = examples["table"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        # answer set
        if 'answers' in examples.keys():
            answers = examples['answers']
        else :
            answers = []

        return (questions, tables, answers)

if __name__=='__main__':
    pass