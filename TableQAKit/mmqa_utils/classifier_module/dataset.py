import json
import gzip
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

MAX_LEN = 512

_CITATION = """\
@article{talmor2021multimodalqa,
  title={MultiModalQA: Complex Question Answering over Text, Tables and Images},
  author={Talmor, Alon and Yoran, Ori and Catav, Amnon and Lahav, Dan and Wang, Yizhong and Asai, Akari and Ilharco, Gabriel and Hajishirzi, Hannaneh and Berant, Jonathan},
  journal={arXiv preprint arXiv:2104.06039},
  year={2021}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the MMQA.
"""

_HOMEPAGE = "https://github.com/allenai/multimodalqa"

_LICENSE = "MIT License"

_TRAINING_FILE = "MMQA_train.jsonl.gz"
_DEV_FILE = "MMQA_dev.jsonl.gz"
_TEST_FILE = "MMQA_test.jsonl.gz"
_TEXTS_FILE = "MMQA_texts.jsonl.gz"
_TABLES_FILE = "MMQA_tables.jsonl.gz"
_PASSAGE_FILE = "MMQA_texts.jsonl.gz"
_IMAGES_INFO_FILE = "MMQA_images.jsonl.gz"
_IMAGES_FILE = "final_dataset_images"
_DATA_PATH = "/home/lfy/lwh/data/mmqa"


class ClassifyDataset(Dataset):

    def __init__(self, tokenizer, type_fn , type_map, data_path = _DATA_PATH , split = 'train') -> None:
        super(ClassifyDataset,self).__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.type_fn = type_fn
        self.type_map = type_map.copy()
        table_path = os.path.join(data_path, _TABLES_FILE)
        passage_path = os.path.join(data_path, _PASSAGE_FILE)
        image_path = os.path.join(data_path, _IMAGES_INFO_FILE)

        tables = {}
        with gzip.open(table_path, 'r') as f: # 表格
            for line in f:
                table = json.loads(line)
                tables[table["id"]] = table
        self.tables = tables

        texts = {}
        with gzip.open(passage_path, 'r') as f: # 文章，段落
            for line in f:
                text = json.loads(line)
                texts[text["id"]] = text
        self.texts = texts
        
        images = {}
        with gzip.open(image_path, 'r') as f: # 图片
            for line in f:
                image = json.loads(line)
                images[image["id"]] = image
        self.images = images
        if split == "train":
            file_path = os.path.join(data_path, _TRAINING_FILE)
        elif split == "dev":
            file_path = os.path.join(data_path, _DEV_FILE)
        elif split == "test":
            file_path = os.path.join(data_path, _TEST_FILE)
        else:
            raise ValueError("Invalid split name")
        
        with gzip.open(file_path, 'r') as f: # 问题
            self.data = [json.loads(line) for line in f]

        for data in tqdm(self.data):
            question_ids = self.tokenizer.encode(data['question'])
            data['input_ids'] = question_ids


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO: add table title, image title, passage title
        data_item = self.data[index]
        meta_data = data_item['metadata']
        qid = data_item['qid']

        if self.split in ["train", "dev"]:
            label = self.type_map[self.type_fn(meta_data['type'])]
            return data_item['input_ids'], label, qid
        else:
            return data_item['input_ids'], None, qid

def collate(data, tokenizer, max_bert_len, test=False):
    bs = len(data) # batchsize
    input_ids, labels, qids = zip(*data)
    input_ids = list(input_ids)

    if not test:
        labels = list(labels)
        labels_tensor = torch.tensor(labels)
        # labels_tensor = torch.zeros(bs,len(LABELS.keys()))
        # for i in range(bs):
        #     labels_tensor[i][labels[i]] = torch.tensor(1)

    max_len = max([len(ids) for ids in input_ids])
    if max_len > max_bert_len:
        max_len = max_bert_len
    for i in range(len(input_ids)):
        input_ids[i] = input_ids[i][:max_len]
        input_ids[i] = input_ids[i] + [tokenizer.pad_token_id] * (max_len - len(input_ids[i]))
    
    input_mask = torch.where(torch.tensor(input_ids) == tokenizer.pad_token_id, torch.tensor(0), torch.tensor(1))
    input_ids = torch.tensor(input_ids)

    if not test:
        return {
            "input_ids": input_ids.cuda(),
            "input_mask": input_mask.cuda(),
            "labels": labels_tensor.cuda(),
            "qids": qids
        }
    else:
        return {
            "input_ids": input_ids.cuda(),
            "input_mask": input_mask.cuda(),
            "qids": qids
        }