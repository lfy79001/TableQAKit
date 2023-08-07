import numpy as np
import torch
import sys
import random
import datetime
import io
from dataclasses import dataclass, field


@dataclass
class ICLArguments:
    random_seed: int = field(default=42)
    resume_id: int = field(default=0)
    max_length: int = field(default=256)
    api_time_interval: float = field(default=1.0)
    temperature: float = field(default=0.1)
    logging_dir: str = field(default='./log')
    output_path: str = field(default='./data/test_predictions.json')
    use_table_markdown: bool = field(default=True)
    use_table_flatten: bool = field(default=False)
    truncation: int = field(default=3000)


class Logger(object):
    def __init__(self, logging_path, stream=io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')):
        self.terminal = stream
        self.log = open(logging_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_time(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now.replace("/", "_").replace(" ", "").replace(":", "_")
    else:
        pass
