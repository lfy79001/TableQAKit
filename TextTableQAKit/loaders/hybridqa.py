import os
import platform

import re
import logging
import json
from TextTableQAKit.structs.data import Cell, Table, HFTabularDataset


logger = logging.getLogger(__name__)

class HybridQA(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_split(self, split):
        question_file_map = {
            "train": "train.json",
            "dev": "dev.json",
            "test": "test.json"
        }

        file_base_path = 'datasets/hybridqa'
        logger.info(f"Loading hybridqa - {split}")
        question_file_path = os.path.join(file_base_path, question_file_map[split])

        with open(question_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)

        data = []

        for question_data in question_data_list:
            properties_info = {}
            question = question_data['question']
            table_id = question_data['table_id']

            if 'question_postag' in question_data:
                properties_info['question--question_postag'] = str(question_data['question_postag'])


            if platform.system() == 'Windows':
                illegal_chars = r'[\\/:\*\?"<>|]'
                table_id = re.sub(illegal_chars, '_', table_id)

            with open(f'{file_base_path}/tables_tok/{table_id}.json', 'r', encoding='utf-8') as f:
                table_data = json.load(f)

            with open('{}/request_tok/{}.json'.format(file_base_path, table_id), 'r', encoding='utf-8') as f:
                requested_document = json.load(f)

            table_data_headers = [aa[0] for aa in table_data['header']]


            table_data_contents = []
            for row in table_data['data']:
                table_data_row_contents = []
                for cell in row:
                    table_data_cell_content = cell[0]
                    for i, link_data in enumerate(cell[1]):
                        if i == 0:
                            table_data_cell_content += '##[HERE STARTS THE HYPERLINKED PASSAGE]##'
                        table_data_cell_content += ("[HYPERLINKED PASSAGE "+str(i+1)+"]: "+requested_document[link_data])
                    table_data_row_contents.append(table_data_cell_content)
                table_data_contents.append(table_data_row_contents)

            if 'url' in table_data:
                properties_info['table--url'] = str(table_data['url'])
            if 'title' in table_data:
                properties_info['table--title'] = str(table_data['title'])
            if 'section_title' in table_data:
                properties_info['table--section_title'] = str(table_data['section_title'])
            # if 'section_text' in table_data:
            #     properties_info['table[section_text]'] = str(table_data['section_text'])
            # if 'intro' in table_data:
            #     properties_info['table[intro]'] = str(table_data['intro'])


            data.append({
                "question": question,
                "table": {
                    "header": table_data_headers,
                    "content": table_data_contents
                },
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
        t.is_linked = True
        t.type = "default"
        t.default_question = entry["question"]

        t.props = entry['properties']

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
    HybridQA().load_split_test("train")