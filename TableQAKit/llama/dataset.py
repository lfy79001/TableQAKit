import json
from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments

from .template import Template
from .utils import DataTrainingArguments, IGNORE_INDEX


def preprocess_data(
        prompt_template: Template,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataTrainingArguments,
        training_args: Seq2SeqTrainingArguments
) -> Dataset:
    column_names = list(dataset.column_names)

    # support question with a single answer or multiple answers
    def get_dialog(examples):
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + "\n" + examples["query"][i] if examples["query"][i] else query
                prefix = examples["prefix"][i] if examples["prefix"][i] else ""
                dialog = prompt_template.get_dialog(query, answer, examples["history"][i], prefix)
                yield dialog

    def preprocess_supervised_dataset(examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for input with history, we build multiple input-label pairs just like:
        # https://github.com/lm-sys/FastChat/blob/f17c092f64840fa6354ed52789dccb2daa793d0b/fastchat/train/train.py#L112
        model_inputs = {"input_ids": [], "labels": []}
        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=False)
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                if len(source_ids) > data_args.max_source_length - 1:  # bos token
                    source_ids = source_ids[:data_args.max_source_length - 1]
                if len(target_ids) > data_args.max_target_length - 1:  # eos token
                    target_ids = target_ids[:data_args.max_target_length - 1]

                input_ids += [tokenizer.bos_token_id] + source_ids + target_ids + [tokenizer.eos_token_id]
                labels += [IGNORE_INDEX] * (len(source_ids) + 1) + target_ids + [tokenizer.eos_token_id]

            if len(input_ids) > data_args.max_source_length + data_args.max_target_length:
                input_ids = input_ids[:data_args.max_source_length + data_args.max_target_length]
            if len(labels) > data_args.max_source_length + data_args.max_target_length:
                labels = labels[:data_args.max_source_length + data_args.max_target_length]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    preprocess_function = preprocess_supervised_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )
        print_supervised_dataset_example(dataset[0])

        return dataset


class LLaMaDataset(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset = self.__read_data_to_huggingface_dataset__(data_path)

    @abstractmethod
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        """
        Reading the data and preprocessing to the huggingface dataset with the column_names bellow:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        :return: dataset: the huggingface Dataset
        """
        pass


class MultiHiertt(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            query = ""
            for text_evidence in one["qa"]["text_evidence"]:
                query += (one["paragraphs"][text_evidence] + "\n")

            for table_evidence in one["qa"]["table_evidence"]:
                query += (one["table_description"][table_evidence] + "\n")
            dataset.append({
                "prefix": None,
                "prompt": "According to the information, use the operation in "
                          "[add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_max, table_min] "
                          "to construct a program to answer the question:" + one["qa"]["question"],
                "query": query,
                "response": one["qa"]["program"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class SKG(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self, data_path: str) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
            dataset.append({
                "prefix": None,
                "prompt": "According to the table, try your best to answer the question: " + one["text_in"],
                "query": one["struct_in"],
                "response": one["seq_out"],
                "history": None
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset
