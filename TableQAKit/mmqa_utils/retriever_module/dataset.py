import json
import gzip
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from utils.image_stuff import get_caption_map

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



class RetrieverDataset(Dataset):
    def __init__(self, tokenizer, data_path = _DATA_PATH , split = 'train', caption_file_name = 'mmqa_captions_llava.json') -> None:
        super(RetrieverDataset,self).__init__()
        self.tokenizer = tokenizer
        self.split = split
        with open(os.path.join(data_path, caption_file_name), "r") as f:
            caption_map = json.load(f)
        self.caption_map = caption_map

        table_path = os.path.join(data_path, _TABLES_FILE)
        passage_path = os.path.join(data_path, _PASSAGE_FILE)
        image_path = os.path.join(data_path, _IMAGES_INFO_FILE)

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
            data['question_ids'] = question_ids

        tables = {}
        with gzip.open(table_path, 'r') as f: # 表格
            for id, line in enumerate(tqdm(f)):
                table = json.loads(line)
                tables[table["id"]] = table
        self.tables = tables

        texts = {}
        with gzip.open(passage_path, 'r') as f: # 文章，段落
            for id, line in enumerate(tqdm(f)):
                text = json.loads(line)
                # text['ids'] = tokenizer.encode(text['text'])
                texts[text["id"]] = text
        self.texts = texts
        
        images = {}
        with gzip.open(image_path, 'r') as f: # 图片
            for id, line in enumerate(tqdm(f)):
                image = json.loads(line)
                # image['ids'] = tokenizer.encode(caption_map[image['id']])
                images[image["id"]] = image
        self.images = images

    def get_caption_by_id(self, id):
        try:
            caption = self.caption_map[id]
        except:
            caption = ""
        return caption

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO: add table title, image title, passage title
        """
        Returns:
            question_ids: List[int]
            image_ids: Dict
            text_ids: Dict
            gold_image: List[str]
            gold_text: List[str]
        """
        data_item = self.data[index]
        meta_data = data_item['metadata']
        question_ids = data_item['question_ids']

        if self.split in ['train','dev']:
            gold_image = []
            gold_text = []
            for sup in data_item['supporting_context']:
                if sup['doc_part'] == 'text':
                    gold_text.append(sup['doc_id'])
                elif sup['doc_part'] == 'image':
                    gold_image.append(sup['doc_id'])
            
            gold_image = list(set(gold_image))
            gold_text = list(set(gold_text)) # unique
        
         
        # text_ids = meta_data['text_doc_ids']
        # image_ids = meta_data['image_doc_ids']

        # image_ids = dict()
        # text_ids = dict()
        # for id in meta_data['text_doc_ids']:
        #     text_ids[id] = self.tokenizer.encode(self.texts['id']['text'])

        # for id in meta_data['image_doc_ids']:
        #     image_ids[id] = self.tokenizer.encode(caption_map[id])
        
        image_docs = dict()
        text_docs = dict()
        for id in meta_data['text_doc_ids']:
            text_docs[id] = {
                'title': self.texts[id]['title'],
                'text': self.texts[id]['text']
            }
        
        for id in meta_data['image_doc_ids']:
            image_docs[id] = {
                'title': self.images[id]['title'],
                'text': self.get_caption_by_id(id)
            }
        
        if self.split in ["train", "dev"]:
            return question_ids, image_docs, text_docs, gold_image, gold_text
        else:
            return question_ids, image_docs, text_docs, None, None


def collate(data, tokenizer, max_bert_len, image_or_text='text', test=False):
    # print(json.dumps(data, indent=4))
    # exit(0)
    data = data[0]
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    question_ids, images_docs, text_docs, gold_image, gold_text = data
    # print(question_ids)
    if image_or_text == 'text':
        docs = text_docs
        gold_docs = gold_text
    else :
        docs = images_docs
        gold_docs = gold_image
    
    row_num = len(docs)
    input_ids = []
    labels = torch.zeros(row_num)
    max_input_length = 0

    for doc_id,item in docs.items():
        input_data = question_ids.copy()  #  Question
        title = item['title']
        text = item['text']
        title_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title)) 
        text_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))  
        input_data.extend(title_ids) # Title
        input_data.append(sep_id) # [SEP]
        input_data.extend(text_ids) # Passage/Image Text
        input_data.append(sep_id) # [SEP]
        max_input_length = max(max_input_length, len(input_data))
        input_ids.append(input_data)

    max_input_length = min(max_input_length, max_bert_len)

    for id,item in enumerate(input_ids):
        if len(item) > max_bert_len:
            item = item[:max_bert_len-1]
            item.append(sep_id)
        else:
            item = item + (max_input_length - len(item)) * [pad_id] # PAD is after SEP
        
        input_ids[id] = item

    
    input_ids = torch.tensor(input_ids)
    input_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    metadata = {
        "gold_docs": gold_docs,
        "docs": [doc_id for doc_id in docs.keys()]
    }
    # print(input_ids.shape)
    # print(input_mask.shape)

    if not test:
        num_gold = len(gold_docs)
        gold_list = [1/num_gold if doc_id in gold_docs else 0 for doc_id in docs.keys()]
        labels = torch.softmax(torch.tensor(gold_list), dim=-1, dtype=torch.float)
        # print(labels)
        return {
            "input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "labels":labels.cuda(), \
            "metadata": metadata
        }
    else:
        return {
            "input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "labels": None, \
            "metadata": metadata
        }