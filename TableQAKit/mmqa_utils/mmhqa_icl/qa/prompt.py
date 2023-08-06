import random
from typing import Dict, Tuple
import pandas as pd
import copy
import os
import json

class PromptBuilder(object):
    def __init__(self, args):
        self.args = args
        with open(os.path.join(args.data_path, args.caption_file_name), "r") as f:
            caption_map = json.load(f)
        self.caption_map = caption_map

    def _passage_prompt(self, passages, db_style_prompt=False):
        """
        Return the passage prompt.
        """
        if len(passages) == 0:
            return ""
        if not db_style_prompt:
            string = "Passages: \n"
            for passage_idx in range(len(passages['id'])):
                string += f"\"{passages['title'][passage_idx]}\"" + f" ({passages['text'][passage_idx]})" + '\n'
            string += '\n'
            return string
        else:
            passage_table_prompt = ""
            _header = []
            _rows = [[]]
            for passage in passages:
                # TODO: add passage prompt
                _header.append(passage['title'])
                _rows[0].append(passage['text'])

            return passage_table_prompt

    def _image_prompt(self, images, db_style_prompt=False):
        """
        Return the image prompt.
        """
        if len(images) == 0:
            return ""
        if not db_style_prompt:
            string = "Images: \n"
            for image_idx in range(len(images['id'])):
                string += f"\"{images['title'][image_idx]}\"" + f" ({self.caption_map[images['id'][image_idx]]})" + '\n'
            string += '\n'
            return string
        else:
            image_table_prompt = ""
            _header = []
            _rows = [[]]
            for image in images:
                # TODO: add image prompt
                _header.append(image['title'])
                _rows[0].append(image['caption'])
            return image_table_prompt

    def table2text_prompt(self, table, drop_row_id=False, ):
        _table = copy.deepcopy(table)
        header = _table['header'][0]
        rows = _table['rows'][0]
        table_title = _table['title'][0]
        if drop_row_id:
            if header[0] == "row_id":
                header = header[1:]
                rows = [_row[1:] for _row in rows]

        prompt_str = 'Table {}: \n'.format(table_title) if table_title else 'Table: \n'
        prompt_str += "\t".join(header) + "\n"
        prompt_str += '\n'.join(["\t".join([str(cell) if len(cell)>0 else '(null)' for cell in row]) for row in rows]) + "\n"
        return prompt_str

    def build_generate_prompt(
            self,
            qtype: Tuple,
            table: Dict = None,
            question: str = None,
            passages: Dict = None,
            images: Dict = None,
            cot: bool = False,
            supporting_context: Dict = None,
            **kwargs
    ):
        """
        Build the prompt of the generation sample.
        """
        generate_prompt = ""
        # task instruction
        if qtype == "image":
            generate_prompt += self._image_prompt(images=images)
            generate_prompt += f"Question: {question}\n"
        elif qtype == "text":
            generate_prompt += self._passage_prompt(passages=passages)
            generate_prompt += f"Question: {question}\n"
        elif qtype == "table":
            generate_prompt += self.table2text_prompt(table=table) + '\n'
            # generate_prompt += f"Question: {question}\n"
            generate_prompt += f"Question: {question}\n" # no type method
        else:
            generate_prompt += self._image_prompt(images=images)
            generate_prompt += self._passage_prompt(passages=passages)
            generate_prompt += self.table2text_prompt(table=table) + '\n'
            # generate_prompt += f"Question: {question}\n"
            generate_prompt += f"Question: {question}\n" # no type method

        if cot:
            if qtype in ['image']:
                generate_prompt += "Please answer the question step by step, you can make reasonable inference to give a definite answer.\n"
            else:
                generate_prompt += f"Please answer the question step by step.\n"
        else:
            generate_prompt += f"Answer: "

        return generate_prompt

