from dataclasses import dataclass, field
import random
from typing import Union, List

import numpy as np
import torch
from transformers import HfArgumentParser

@dataclass
class RunnningArguments:
    data_path: str = field(
        metadata={"help": "The path of mmqa dataset."}
    )
    caption_file_name: str=field(
        default='mmqa_captions_llava.json',
        metadata={"help": "Caption file name"}
    )
    split: str=field(
        default='validation',
        metadata= {
            "help": "run on which set, choices = ['train', 'validation', 'test']"
        }
    )
    retriever_files_path: str=field(
        default="utils/retriever_files/",
        metadata={
            "help": "TYPE_FILE: question_type_\{SPLIT\}.json"
                    "QIM_FILE: question_image_match_scores_\{SPLIT\}.json"
                    "QPM_FILE: question_passage_match_scores_\{SPLIT\}.json"
        }
    )
    top_n: int=field(
        default=3,
        metadata={
            "help": "use top_n retrieved files"
        }
    )
    template_path: str=field(
        default="mmhqa_icl/templates",
        metadata= {
            "help": "Path to the directory of templates"
        }
    )
    prompt_file: str=field(
        default="prompt.json",
        metadata={
            "help": "Name of the config file of icl"
        }
    )
    api_keys_file: str=field(
        default=None,
        metadata= {
            "help": "Path to the api_keys file"
        }
    )
    save_dir: str=field(
        default='results/',
        metadata= {
            "help": "Path to save directory"
        }
    )
    save_file_name: str=field(
        default=None,
    )

    # Multiprocess options
    n_processes: int=field(
        default=1,
    )

    # Generation options
    seed: int=field(
        default=42,
    )
    resume: bool=field(default=False, metadata={"help": "whether to resume from last result file"})
    oracle_classifier: bool=field(default=False, metadata={"help": "whether to use oracle classifier"})
    oracle_retriever: bool=field(default=False, metadata={"help": "whether to use oracle retriever"})

    # LLM options
    n_parallel_prompts: int=field(default=1)
    max_api_total_tokens: int=field(default=4001)
    temperature: float=field(default=0.4, metadata={"help": "temperature param of LLM"})
    engine: str=field(
        default="text-davinci-003",
        metadata={"help": "choices= ['text-davinci-003']"}
    )
    top_p: float=field(
        default=1.0
    )
    stop_tokens: str=field(
        default='\n\n\n',
        metadata={
            "help": 'Split stop tokens by ||'
        }
    )
    # debug options
    verbose: bool=field(
        default=False,
        metadata={
            "help": "debug options"
        }
    )

def get_running_args():
    return HfArgumentParser(
        RunnningArguments
    ).parse_args_into_dataclasses()[0]