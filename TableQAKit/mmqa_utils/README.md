## MMQA utils



### Introduction

Folder Structure：

```
└─mmqa_utils
    └─classifier_module
    	├─__init__.py
    	├─dataset.py
    	├─model.py
    	├─trainer.py
    	├─utils.py
    └─retriever_module
    	├─__init__.py
    	├─dataset.py
    	├─model.py
    	├─trainer.py
    	├─utils.py
    └─mmhqa_icl             # mmhqa-icl framework
        └─templates         # folder containing icl demos
        └─retriever_files   # folder containing classifying and retrieving results
        └─qa				# folder containing QA modules
        	├─__init__.py
        	├─openai_qa.py
        	├─prompt.py
	    ├─__init__.py
    	├─args.py
    	├─mmqa.py           # dataset builder
    	├─main.py
    	├─utils.py
```



**data preparation:**

download MultimodalQA dataset and caption files from our [Huggingface](https://huggingface.co/datasets/TableQAKit/MMQA/tree/main)



### Classifier

run  `ClassifierTrainer` 

```bash
CUDA_VISIBLE_DEVICES=0 python test_classifier.py \
--output_dir ./output \
--bert_model <path-to-bert_model> \
--per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
--data_path <path-to-data-folder>
```



Demo code:

```python
# test_classifier.py
from TableQAKit.mmqa_utils.classifier_module import ClassifierTrainer
trainer = ClassifierTrainer()
trainer.train()
```



### Retriever

run  `RetrieverTrainer` 

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python test_retriever.py \
--output_dir ./output \
--bert_model <path-to-bert_model> \
--data_path <path-to-data-folder> \
--n_gpu 3 \
--image_or_text image
```



Demo Code:

```python
# test_retriever.py
from TableQAKit.mmqa_utils.retriever_module import RetrieverTrainer
trainer = RetrieverTrainer()
trainer.train()
```



**Tips:**

- set `--top_n` to select different number of retrieved images or passages
- set `--image_or_text` to `image` or `text` to train an image retriever or a passage retriever
- set `--num_train_epochs` to train more epochs



### MMHQA-ICL

Full Code will be released [here](https://github.com/NeosKnight233/MMHQA-ICL).



run `MMHQA-ICL`

```
python test_mmhqa_icl.py \
--data_path <path-to-data-folder> \
--retriever-files-path <path-to-retriever_files-folder> \
--template_path <path-to-templates-folder>
```



Demo Code:

```python
# test_mmhqa_icl.py
from TableQAKit.mmqa_utils.mmhqa_icl import MMQA_Runner
runner = MMQA_Runner(api_keys=['<key1>'])
runner.predict()
```



**Tips:**

- **You need to set `_DATA_PATH` in `mmqa.py` to your local dataset** 

- set `--top_n` to select different number of retrieved images or passages
- set `--template_path` to path to local icl demos file
- set `--retriever_files_path` to path to results of classifier and retriever
- set `--oracle_classifier` and `--oracle_retriever` to use golden results of classifier and retriever
