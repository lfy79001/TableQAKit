import os
import platform

import re
import logging
import json
from structs.data import Cell, Table, HFTabularDataset


logger = logging.getLogger(__name__)

class TATQA(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_split(self, split):
        question_file_map = {
            "train": "train.json",
            "dev": "dev.json",
            "test": "test.json"
        }

        file_base_path = 'datasets/TATQA'
        logger.info(f"Loading TAT-QA - {split}")
        question_file_path = os.path.join(file_base_path, question_file_map[split])

        with open(question_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)

        data = []

        for question_data in question_data_list:

            properties_info = {}

            table_data = question_data["table"]["table"]
            table_data_headers = table_data[0]
            table_data_contents = table_data[1:]
            txt_list = [paragraph["text"] for paragraph in question_data["paragraphs"]]

            for single_question_data in question_data["questions"]:
                question = single_question_data["question"]
                data.append({
                    "question": question,
                    "table": {
                        "header": table_data_headers,
                        "content": table_data_contents
                    },
                    "txt": txt_list,
                    "properties": properties_info
                })
        self.data[split] = data
        self.dataset_info = {
            "citation": "111",
            "description": "111",
            "version": "111",
            "license": "111"
        }

    def load_split_test(self, split):
        self._load_split(split)

    def prepare_table(self, entry):
        t = Table()
        t.type = "default"
        t.default_question = entry["question"]

        t.props = entry['properties']
        t.txt_info = entry['txt']

        for header_cell in entry["table"]["header"]:
            c = Cell()
            c.value = header_cell
            c.is_col_header = True
            t.add_cell(c)
        t.save_row()

        for row in entry["table"]["content"]:
            for cell in row:
                c = Cell()
                c.value = cell
                t.add_cell(c)
            t.save_row()
        return t
if __name__ == '__main__':
    TATQA().load_split_test("train")