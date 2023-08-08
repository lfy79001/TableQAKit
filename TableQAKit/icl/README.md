# TextTableQAKit/In-Context-Learning

## QuickStart

### Using turbo directly
```
import json
import os
from typing import List, Union, Dict
from TableQAKit.icl import turbo, MultiHiertt, turboICL
data_path = "path to your data"
demo_path = "path to your demonstration or None"

dataset = MultiHiertt(data_path=data_path, demo_path=demo_path)
model = turbo(key="your openai key")
infer = turboICL(model=model, dataset=dataset)
infer.infer(demo_prefix=None, cot_trigger=None, answer_trigger="Therefore, the answer is ")
```

### using davinci_003 directly
```
import json
import os
from typing import List, Union, Dict
from TableQAKit.icl import text_davinci_003, MultiHiertt, davinciICL
data_path = "path to your data"
demo_path = "path to your demonstration or None"


dataset = MultiHiertt(data_path=data_path, demo_path=demo_path)
model = text_davinci_003(key="your openai key")
infer = davinciICL(model=model, dataset=dataset)
infer.infer(demo_prefix=None, cot_trigger=None, answer_trigger="Therefore, the answer is ")
```

### Using turbo for CoT
```
import json
import os
from typing import List, Union, Dict
from TableQAKit.icl import turbo, MultiHiertt, turboICL
data_path = "path to your data"
demo_path = "path to your demonstration or None"


dataset = MultiHiertt(data_path=data_path, demo_path=demo_path)
demo_prefix = "Reading the texts and tables and try your best to answer the question."
model = turbo(key="your openai key")
infer = turboICL(model=model, dataset=dataset)
infer.infer(demo_prefix=demo_prefix, cot_trigger="Let's think step by step.", answer_trigger="Therefore, the answer is ")
```

### using davinci_003 for CoT
```
import json
import os
from typing import List, Union, Dict
from TableQAKit.icl import text_davinci_003, MultiHiertt, davinciICL
data_path = "path to your data"
demo_path = "path to your demonstration or None"


dataset = MultiHiertt(data_path=data_path, demo_path=demo_path)
demo_prefix = "Reading the texts and tables and try your best to answer the question."
model = text_davinci_003(key="your openai key")
infer = davinciICL(model=model, dataset=dataset)
infer.infer(demo_prefix=demo_prefix, cot_trigger="Let's think step by step.", answer_trigger="Therefore, the answer is ")
```

### Using turbo for PoT
```
import json
import os
from typing import List, Union, Dict
from TableQAKit.icl import turbo, MultiHiertt, turboICL
data_path = "path to your data"
demo_path = "path to your demonstration or None"


dataset = MultiHiertt(data_path=data_path, demo_path=demo_path)
demo_prefix = "Read the following text and table, and then write code to answer a question:"
model = turbo(key="your openai key")
infer = turboICL(model=model, dataset=dataset)
infer.infer(demo_prefix=demo_prefix, cot_trigger=None, answer_trigger="#Python\n")
```

### using davinci_003 for PoT
```
import json
import os
from typing import List, Union, Dict
from TableQAKit.icl import text_davinci_003, MultiHiertt, davinciICL
data_path = "path to your data"
demo_path = "path to your demonstration or None"


dataset = MultiHiertt(data_path=data_path, demo_path=demo_path)
demo_prefix = "Read the following text and table, and then write code to answer a question:"
model = text_davinci_003(key="your openai key")
infer = davinciICL(model=model, dataset=dataset)
infer.infer(demo_prefix=demo_prefix, cot_trigger=None, answer_trigger="#Python\n")
```

### Create New Dataset
```
from TableQAKit.icl import GPTDataSet


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