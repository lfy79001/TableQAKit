import os
import platform

import re
import logging
import json
from structs.data import Cell, Table, HFTabularDataset


logger = logging.getLogger(__name__)

class SpreadSheetQA(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_split(self, split):
        question_file_map = {
            "train": "train.json",
            "dev": "dev.json",
            "test": "test.json"
        }

        file_base_path = 'datasets/SpreadSheetQA'
        logger.info(f"Loading SpreadSheetQA - {split}")
        question_file_path = os.path.join(file_base_path, question_file_map[split])

        with open(question_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)

        data = []

        for question_data in question_data_list:

            properties_info = {}

            if 'explanation' in question_data['qa']:
                properties_info['question--explanation'] = str(question_data['qa']['explanation'])
            if 'program' in question_data['qa']:
                properties_info['question--program'] = str(question_data['qa']['program'])
            if 'program_re' in question_data['qa']:
                properties_info['question--program_re'] = str(question_data['qa']['program_re'])
            if 'source' in question_data:
                properties_info['source'] = str(question_data['source'])

            question = question_data['qa']['question']
            if type(question_data['table_ori']) == list:
                table_data = question_data['table_ori']
            else:
                table_data = question_data['table_ori']['table']
            table_data_headers = table_data[0]
            table_data_contents = table_data[1:]
            txt_list = question_data['pre_text'] + question_data['post_text']
            txt_list = list(filter(lambda x: x not in ['.', '*'], txt_list))


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
            for (i, cell) in enumerate(row):
                c = Cell()
                if i == 0:
                    c.is_row_header = True
                c.value = cell
                t.add_cell(c)
            t.save_row()
        return t
if __name__ == '__main__':
    SpreadSheetQA().load_split_test("train")