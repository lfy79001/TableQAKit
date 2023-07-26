# TextTableQAKit/llama

## QuickStart

### MultiHiertt Dataset as a demonstration
```
from llama import LLaMaTrainer
from llama import MultiHiertt
from llama import Template


class Trainer(LLaMaTrainer):
    def __getTemplate__(self) -> Template:
        template = Template()
        return template


datasets = MultiHiertt()
trainer = Trainer([datasets])
trainer.train()
```

### Add new datasets
```
import json
from datasets import Dataset
from llama import LLaMaTrainer
from llama import LLaMaDataset
from llama import Template


class NewDataset(LLaMaDataset):
    def __read_data_to_huggingface_dataset__(self) -> Dataset:
        column_names = ["prefix", "prompt", "query", "response", "history"]
        data = json.load(
            open(your_data_path, 'r', encoding='utf-8')
        )
        dataset = []
        for one in data:
			# process your data to:
            dataset.append({
                "prefix": prefix,
                "prompt": prompt,
                "query": query,
                "response": response,
                "history": history
            })

        huggingface_data = {column_name: [] for column_name in column_names}

        for data_sample in dataset:
            for column_name in column_names:
                huggingface_data[column_name].append(data_sample[column_name])

        hf_dataset = Dataset.from_dict(huggingface_data)
        return hf_dataset


class NewTemplate(Template):
    def __post_init__(self):
        self._register_template(
            prefix="A chat between a curious user and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            prompt="Human: {query}\nAssistant: ",
            sep="\n",
            use_history=False
        )


class Trainer(LLaMaTrainer):
    def __getTemplate__(self) -> Template:
        template = NewTemplate()
        return template


datasets = NewDataset()
trainer = Trainer([datasets])
trainer.train()
```