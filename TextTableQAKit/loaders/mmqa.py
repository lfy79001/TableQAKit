import os

import datasets
import logging
import json
from structs.data import Cell, Table, HFTabularDataset


logger = logging.getLogger(__name__)

class MultiModalQA(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_split(self, split):

        question_file_map = {
            "train" : "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_train.jsonl",
            "dev" : "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_dev.jsonl",
            "test" : "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_test.jsonl"
        }
        logger.info(f"Loading multimodalqa - {split}")


        '''
        loading dataset
        '''
        # loading txt
        txt_file_path = "datasets/mmqa/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl"
        txt_info = {}

        with open(txt_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                id_value = json_data['id']
                text_value = json_data['text']
                txt_info[id_value] = text_value

        # print(txt_info)

        #loading pic
        pic_file_path = "datasets/mmqa/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl"
        pic_info = {}

        with open(pic_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                id_value = json_data['id']
                path_value = json_data['path']
                pic_info[id_value] = path_value

        # print(pic_info)

        #loading table
        table_file_path = "datasets/mmqa/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl"
        table_info = {}
        table_meta_info = {}

        with open(table_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                id_value = json_data['id']
                table_value = json_data['table']
                table_rows_value = table_value["table_rows"]
                table_rows_text_value = []
                for rows in table_rows_value:
                    temp_list = []
                    for col in rows:
                        temp_list.append(col["text"])
                    table_rows_text_value.append(temp_list)
                header_value = table_value["header"]
                header_text_value = []
                for row in header_value:
                    header_text_value.append(row["column_name"])
                table_info[id_value] = {}
                table_meta_info[id_value] = {}
                table_info[id_value]["table_rows_text"] = table_rows_text_value
                table_info[id_value]["header_text"] = header_text_value
                if 'title' in json_data:
                    table_meta_info[id_value]["title"] = json_data['title']
                if 'url' in json_data:
                    table_meta_info[id_value]["url"] = json_data['url']
                if 'table_name' in json_data['table']:
                    table_meta_info[id_value]["table_name"] = json_data['table']['table_name']
        # print(table_info)

        # loading question
        question_file_name = question_file_map[split]
        question_file_path = os.path.join("datasets/mmqa", question_file_name)
        question_info = []
        # print(question_file_path)

        with open(question_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                properties_info = {}
                json_data = json.loads(line)
                # qid_value = json_data['qid']
                question_value = json_data['question']
                image_ids_value = json_data['metadata']['image_doc_ids']
                txt_ids_value = json_data['metadata']['text_doc_ids']
                table_id_value = json_data['metadata']["table_id"]
                if 'type' in json_data['metadata']:
                    properties_info['question--type'] = str(json_data['metadata']['type'])
                if 'modalities' in json_data['metadata']:
                    properties_info['question--modalities'] = str(json_data['metadata']['modalities'])
                if 'wiki_entities_in_answers' in json_data['metadata']:
                    properties_info['question--wiki_entities_in_answers'] = str(json_data['metadata']['wiki_entities_in_answers'])
                if 'wiki_entities_in_question' in json_data['metadata']:
                    properties_info['question--wiki_entities_in_question'] = str(json_data['metadata']['wiki_entities_in_question'])
                if 'pseudo_language_question' in json_data['metadata']:
                    properties_info['question--pseudo_language_question'] = str(json_data['metadata']['pseudo_language_question'])
                for key in table_meta_info[table_id_value]:
                    properties_info[f'table--{key}'] = str(table_meta_info[table_id_value][key])
                question_info.append({
                    "question" : question_value,
                    "image" : [pic_info[i] for i in image_ids_value],
                    "txt" : [txt_info[i] for i in txt_ids_value],
                    "table": table_info[table_id_value],
                    "properties" : properties_info
                })

        self.dataset_info = {
            "citation" : "111",
            "description" : "111",
            "version" : "111",
            "license" : "111"
        }
        self.data[split] = question_info

    def load_split_test(self, split):
        self._load_split(split)

    def prepare_table(self, entry):
        t = Table()
        t.type = "default"
        t.default_question = entry["question"]
        t.pic_info = entry["image"]
        t.txt_info = entry["txt"]

        t.props = entry['properties']

        for header_cell in entry["table"]["header_text"]:
            c = Cell()
            c.value = header_cell
            c.is_col_header = True
            t.add_cell(c)
        t.save_row()

        for row in entry["table"]["table_rows_text"]:
            for cell in row:
                c = Cell()
                c.value = cell
                t.add_cell(c)
            t.save_row()
        return t
if __name__ == '__main__':
    MultiModalQA().load_split_test("train")