## StructuredQAKit



### Introduction

This is a toolkit of StructuredQA for datasets like wikisql, wikitq, sqa and so on.

Folder Structure：

```
└─stucturedqa
    └─outputs   # default directory to store the results
    └─builder   # dataset builder
    	├─hybridqa.py
        ├─msr_sqa.py
        ├─wikisql_tapas.py
        ├─wikisql.py
        ├─wikitq_tapas.py
        ├─wikitq.py
    └─utils
        ├─common.py
        ├─dataset.py
        ├─tapas_utils.py
        ├─tapas_wikisql_utils.py
        ├─tapex_wikisql_utils.py
    └─stucturedqakit.py
```



#### Model List



| model_name | finetuned_dataset    |
| ---------- | -------------------- |
| tapas      | sqa, wikisql, wikitq |
| tapex      | sqa, wikisql, wikitq |
| reastap    | sqa, wikisql, wikitq |
| omnitab    | wikisql, wikitq      |



#### Dataset List



| Dataset  |
| -------- |
| SQA      |
| wikisql  |
| wikitq   |
| hybridqa |



### Get Started



Demo Code：

```python
from TableQAKit.structuredqa import (
    ToolBaseForStructuredQA,
    BartForStructuredQA,
    BertForStructuredQA
    )
from TableQAKit.structuredqa.utils.dataset import (
    WikisqlDataset, 
    WikitablequestionsDataset, 
    MSR_SQADataset,
    StructuredQADatasetFromBuilder_Tapas,
    StructuredQADatasetFromBuilder
    )

#### BART-based model ####
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/reastap-large')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/reastap-large-finetuned-wtq')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/reastap-large-finetuned-wikisql')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/omnitab-large-finetuned-wtq')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/omnitab-large-finetuned-wikisql')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/microsoft/tapex-large-finetuned-wtq')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/microsoft/tapex-large-finetuned-wikisql')
# tool = BartForStructuredQA(model_name_or_path = '/home/lfy/PTM/tapex-large-finetuned-sqa')

#### BERT-based model (tapas) ####
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large')
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large-finetuned-sqa')
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large-finetuned-wikisql-supervised')
# tool = BertForStructuredQA(model_name_or_path='/home/lfy/PTM/google/tapas-large-finetuned-wtq')



"""select dataset"""
### BART ###
# dataset = WikisqlDataset()
# dataset = WikitablequestionsDataset()
# dataset = MSR_SQADataset()
# dataset = StructuredQADatasetFromBuilder('wikitq')
# dataset = StructuredQADatasetFromBuilder('hybridqa')

### BERT(tapas) ###
# dataset = StructuredQADatasetFromBuilder_Tapas('wikisql_tapas')
# dataset = StructuredQADatasetFromBuilder_Tapas('wikitq_tapas')
# dataset = MSR_SQADataset()


"""eval / predict"""
# tool.eval(dataset=dataset, per_device_eval_batch_size = 8)
# tool.predict(dataset= dataset)
# tool.train(dataset=dataset)

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python -m torch.distributed.launch --nproc_per_node 8 test.py
WANDB_DISABLED="true", CUDA_VISIBLE_DEVICES=3 python test.py
"""
# tool.train(dataset=dataset, per_device_train_batch_size = 4)
```



Tips：

- `BaseStructuredQADataset` is used to build the dataset, which needs to be inherited when using it, and implement the `get_example` method to return (question, form, answer) triplet
- Bert (Tapas) or Bart based models can be invoked as needed



### Metrics Result



| Dataset | Model                                                        | Denotation Acc                                       |
| ------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| WikiSQL | tapas-large-finetuned-wikisql-supervised                     | 0.7028   **(cell selection accuracy)**               |
| WikiSQL | tapex-large-finetuned-wikisql                                | 0.8947                                               |
| WikiSQL | **reastap-large-finetuned-wikisql**                          | **0.8956**                                           |
| WikiSQL | [omnitab-large-finetuned-wikisql](https://huggingface.co/yilunzhao/omnitab-large-finetuned-wikisql) | 0.8874                                               |
|         |                                                              |                                                      |
| Wikitq  | tapas-large-finetuned-wtq                                    | 0.4771   **(cell selection accuracy, 1855 samples)** |
| Wikitq  | tapex-large-finetuned-wtq                                    | 0.5722                                               |
| Wikitq  | reastap-large-finetuned-wtq                                  | 0.5973                                               |
| Wikitq  | **omnitab-large-finetuned-wtq**                              | **0.6097**                                           |
|         |                                                              |                                                      |
| SQA     | tapas-large-finetuned-sqa                                    | 0.521    **(cell selection accuracy)**               |
| SQA     | tapex-large-finetuned-sqa                                    | 0.3987                                               |
| SQA     | *tapex-large-finetuned-wtq*                                  | 0.3775                                               |
| SQA     | *reastap-large-finetuned-wtq*                                | 0.3854                                               |
| SQA     | ***omnitab-large-finetuned-wtq***                            | **0.4146**                                           |

