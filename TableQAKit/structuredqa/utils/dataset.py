import os
import torch
from torch.utils.data import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from transformers import HfArgumentParser
from datasets import load_dataset
from typing import Tuple, List, Optional
import pandas as pd
from .tapex_wikisql_utils import _TYPE_CONVERTER, retrieve_wikisql_query_answer_tapas
from .common import convert_table_types

INTRINSIC_DATASETS = ['wikisql','wikitablequestions']
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
BUILDER_PATH = os.path.join(ROOT_DIR, 'builder')

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


# class BaseStructuredQADataset(Dataset):
class BaseStructuredQADataset(object):
    """
    BaseDataset for Table-Text Question Answering tasks
    """
    def __init__(self, dataset_name, args=None, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.datasets = {
            'train': None,
            'validation': None,
            'test': None
        }
        self.dataset_name = dataset_name

    @property
    def train_dataset(self):
        if 'train' in self.datasets:
            return self.datasets['train']
        return None

    @property
    def eval_dataset(self):
        if 'validation' in self.datasets:
            return self.datasets['validation']
        return None
        

    @property
    def test_dataset(self):
        if 'test' in self.datasets:
            return self.datasets['test']
        return None
    
    @property
    def name(self):
        return self.dataset_name

    @staticmethod
    def get_example(examples) -> tuple:
        """
        a function to get (question, table, answer, ...)
        """
        raise NotImplementedError


class WikisqlDataset(BaseStructuredQADataset):

    def __init__(self, dataset_name='wikisql', dataset_config_name=None, cache_dir=None, **kwargs) -> None:
        super().__init__(dataset_name=dataset_name,**kwargs)
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

class WikitablequestionsDataset(BaseStructuredQADataset):

    def __init__(self, dataset_name='wikitablequestions', cache_dir=None, **kwargs) -> None:
        super().__init__(dataset_name=dataset_name, **kwargs)
        self.datasets = load_intrinsic_dataset(dataset_name, cache_dir=cache_dir)

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

class MSR_SQADataset(BaseStructuredQADataset):

    def __init__(self, cache_dir=None, **kwargs) -> None:
        super().__init__(dataset_name='sqa', **kwargs)
        self.datasets = load_dataset(os.path.join(BUILDER_PATH, 'msr_sqa.py'), cache_dir=cache_dir)
    
    @staticmethod
    def sqa_get_constructed_history_and_golden_response(question_and_history):
        """"""
        if len(question_and_history)<2:
            return question_and_history[-1].strip(), question_and_history[-1].strip()
        else:
            reversed_utterance_head = [question.strip() for question in reversed(question_and_history[:-1])]
            reversed_utterance_head_str = " | ".join(reversed_utterance_head)
            return question_and_history[-1].strip() + " || " + reversed_utterance_head_str, question_and_history[-1]

    @staticmethod
    def get_example(examples) -> tuple:
        questions = [MSR_SQADataset.sqa_get_constructed_history_and_golden_response(question_and_history)[0] for question_and_history in examples["question_and_history"]]
        example_tables = examples["table"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        # answer set
        if 'answer_text' in examples.keys():
            answers = examples['answer_text']
        elif 'answers' in examples.keys():
            answers = examples['answers']
        else :
            answers = []

        if 'answer_coordinates' in examples.keys():
            answer_coordinates = examples["answer_coordinates"]
        else:
            answer_coordinates = []

        return (questions, tables, answers, answer_coordinates)


class StructuredQADatasetFromBuilder(BaseStructuredQADataset):
    """
    Support `wikisql` | `wikisql_tapas` |`wikitq` | `hybridqa` |...

    If you want to use `SQA` dataset, please use `MSR_SQADataset` directly since it is a conversation-based dataset.
    """
    def __init__(self, dataset_name='wikitq', cache_dir=None, **kwargs) -> None:
        """
        Params:
            dataset_name (`str` or `os.PathLike`):
                name of dataset builder, used to create the dataset.
                optional: `wikisql` | `wikisql_tapas` |`wikitq` | `hybridqa`.
        """
        super().__init__(dataset_name=dataset_name, **kwargs)
        if isinstance(dataset_name, str):
            self.datasets = load_dataset(os.path.join(BUILDER_PATH, f'{dataset_name}.py'), cache_dir)
        else:
            self.datasets = load_dataset(dataset_name, cache_dir)

    @staticmethod
    def get_example(examples) -> tuple:
        questions = [question.lower() for question in examples["question"]]
        example_tables = examples["table"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        # answer set
        if 'answer_text' in examples.keys():
            answers = examples['answer_text']
        elif 'answers' in examples.keys():
            answers = examples['answers']
        else :
            answers = []

        return (questions, tables, answers)


class StructuredQADatasetFromBuilder_Tapas(BaseStructuredQADataset):
    # TODO: add tapas support
    """
    Support `wikisql` | `wikisql_tapas` |`wikitq` | `hybridqa` |...

    NOTE: `SQA` dataset is not suppoted, please use `MSR_SQADataset` instead.
    """
    def __init__(self, dataset_name='wikitq', cache_dir=None, **kwargs) -> None:
        """
        Params:
            dataset_name (`str` or `os.PathLike`):
                name of dataset builder, used to create the dataset.
                optional: `wikisql` | `wikisql_tapas` |`wikitq` | `hybridqa`
        """
        super().__init__(dataset_name=dataset_name, **kwargs)
        if isinstance(dataset_name, str):
            self.datasets = load_dataset(os.path.join(BUILDER_PATH, f'{dataset_name}.py'), cache_dir)
        else:
            self.datasets = load_dataset(dataset_name, cache_dir)

    @staticmethod
    def get_example(examples) -> tuple:
        questions = [question.lower() for question in examples["question"]]
        example_tables = examples["table"]
        tables = [
            pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            for example_table in example_tables
        ]
        # answer set
        if 'answer_text' in examples.keys():
            answers = examples['answer_text']
        elif 'answers' in examples.keys():
            answers = examples['answers']
        else :
            answers = []

        if 'answer_coordinates' in examples.keys():
            answer_coordinates = examples["answer_coordinates"]
        else:
            answer_coordinates = []


        return (questions, tables, answers, answer_coordinates)


if __name__=='__main__':
    pass