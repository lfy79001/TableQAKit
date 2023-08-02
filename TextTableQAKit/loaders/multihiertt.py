import os
import platform

import re
import logging
import json
from structs.data import Cell, Table, HFTabularDataset
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class MultiHiertt(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.html = []

    def wrap_with_div(self, table_str):
        return f'<div>{table_str}</div>'

    def _load_split(self, split):
        question_file_map = {
            "train": "train.json",
            "dev": "dev.json",
            "test": "test.json"
        }

        file_base_path = 'datasets/multihiertt'
        logger.info(f"Loading multihiertt - {split}")

        question_file_path = os.path.join(file_base_path, question_file_map[split])

        with open(question_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)

        data = []

        for question_data in question_data_list:
            properties_info = {}

            if 'program' in question_data['qa']:
                properties_info['program'] = str(question_data['qa']['program'])
            if 'question_type' in question_data['qa']:
                properties_info['question_type'] = str(question_data['qa']['question_type'])

            question = question_data['qa']['question']
            
            html_data = ""
            for table_html in question_data["tables"]:

                soup = BeautifulSoup(table_html, 'html.parser')

      
                for table in soup.find_all('table'):
                    table['class'] = "table table-sm no-footer table-bordered caption-top main-table"

           
                updated_html_string = soup.prettify()


                html_data += self.wrap_with_div(updated_html_string)
            
            txt_list = question_data['paragraphs']
            
            data.append({
                "question": question,
                "html": html_data,
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
        t.html = entry["html"]
        
        return t
if __name__ == '__main__':
    MultiHiertt().load_split_test("train")