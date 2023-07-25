import os
import platform

import re
import logging
import json
from structs.data import Cell, Table, HFTabularDataset


logger = logging.getLogger(__name__)

class HiTab(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_all_coordinates(self, json_data):
        coordinates = []

        def dfs(node):
            row_index = node.get("row_index", 0) 
            col_index = node.get("column_index", 0)
            coordinates.append((row_index, col_index))

            for child in node.get("children", []):
                dfs(child)

        dfs(json_data)
        if (-1,-1) in coordinates:
             coordinates.remove((-1,-1))
        return coordinates    

    def _load_split(self, split):
        question_file_map = {
            "train": "train.jsonl",
            "dev": "dev.jsonl",
            "test": "test.jsonl"
        }

        file_base_path = 'datasets/hitab'
        logger.info(f"Loading hitab - {split}")
        question_file_path = os.path.join(file_base_path, question_file_map[split])



        table_data_dict = {}
        for filename in os.listdir(os.path.join(file_base_path, "tables")):
            if filename.endswith(".json"):
                file_path = os.path.join(file_base_path, "tables", filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    table_data_dict[filename[:-5]] = content  


        data = []

        with open(question_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                question_data = json.loads(line)
                properties_info = {}

                if 'table_source' in question_data:
                    properties_info['table_source'] = str(question_data['table_source'])
                if 'aggregation' in question_data:
                    properties_info['aggregation'] = str(question_data['aggregation'])
                
                question = question_data["question"]

                table_data = table_data_dict[question_data["table_id"]]
                
                if 'title' in table_data:
                    properties_info['table_title'] = str(table_data['title'])
                if 'top_header_rows_num' in table_data:
                    properties_info['top_header_rows_num'] = str(table_data['top_header_rows_num'])
                if 'left_header_columns_num' in table_data:
                    properties_info['left_header_columns_num'] = str(table_data['left_header_columns_num'])



                table_data_text = table_data["texts"]

                col_header_idx = self.get_all_coordinates(table_data['top_root']) 
                row_header_idx = self.get_all_coordinates(table_data['left_root'])

                data.append({
                    "question": question,
                    "table": {
                        "text": table_data_text,
                        "col_header": col_header_idx,
                        "row_header": row_header_idx
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
        t.type = "default"
        t.default_question = entry["question"]

        t.props = entry['properties']
       

        for (i,row) in enumerate(entry["table"]["text"]):
            for (j,cell) in enumerate(row):
                c = Cell()
                c.value = cell
                c.is_col_header = (i,j) in entry["table"]["col_header"]
                c.is_row_header = (i,j) in entry["table"]["row_header"]
                t.add_cell(c)
            t.save_row()

        return t
if __name__ == '__main__':
    HiTab().load_split_test("train")