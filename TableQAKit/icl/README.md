# TextTableQAKit/In-Context-Learning

## QuickStart

### Using turbo for MultiHiertt
```
import json
import os
from typing import List, Union, Dict
from icl import turbo, MultiHiertt, turboICL


dataset = MultiHiertt(data_path="./data/val.json", demo_path=None)
demo_prefix = "Reading the texts and tables and try your best to answer the question."
model = turbo(key="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")
infer = turbo(model=model, dataset=dataset)
infer.infer(demo_prefix=demo_prefix, cot_trigger="Let's think step by step.", answer_trigger="Therefore, the answer is ")
```

### using davinci_003 for MultiHiertt
```
import json
import os
from typing import List, Union, Dict
from icl import text_davinci_003, MultiHiertt, davinciICL


dataset = MultiHiertt(data_path="./data/val.json", demo_path=None)
demo_prefix = "Reading the texts and tables and try your best to answer the question."
model = text_davinci_003(key="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")
infer = davinci(model=model, dataset=dataset)
infer.infer(demo_prefix=demo_prefix, cot_trigger="Let's think step by step.", answer_trigger="Therefore, the answer is ")
```

### Create New Dataset
```
from icl import GPTDataSet


class NewDataset(GPTDataSet):
    def read_data(self, data_path: str) -> List[Dict]:
        """
        Read the data from the specified data path and preprocess it into the following format:
        :param data_path: The path of the data file.
        :return: A list of dictionaries, each containing the following keys:
            - "id": The unique identifier of the data.
            - "question": The question associated with the data.
            - "texts": A list of strings representing the texts associated with the data.
            - "rows": A list of strings representing the tables' rows associated with the data.
        """
        pass

    def read_demos(self, demo_path: str) -> List[Dict]:
        """
        Read the demonstration data from the specified demo path and preprocess it into the following format:
        :param demo_path: The path of the demonstration file.
        :return: A list of dictionaries, each containing the following keys:
            - "question": The question of the demonstration.
            - "rationale": The CoT (Causes of Truth) rationale of the demonstration.
            - "answer": The golden answer of the demonstration.
        """
        pass
```

### Start infer
```
python main.py \
--random_seed 42 \
--resume_id 0 \
--max_length 256 \
--api_time_interval 1.0 \
--temperature 0.1 \
--logging_dir ./log \
--output_path ./data/test_predicitions.json \
--use_table_markdown True \
--use_table_flatten False \
--truncation 2048
```