## StructuredQAKit



### Introduction



以 `ReasTAP` 的代码为基础改写。部分代码改写自 `tapas_utils`  与 `UnifiedSKG`



- 新增 `BaseStructuredQADataset` 等类，使用时需要继承该类，需要用户实现 get_example 方法，用于返回 (问题，表格，答案) 三元组
- 一些预置 Dataset，用户可自己编写额外的数据集类
- 根据需要调用基于 Bert (Tapas) 或 Bart 的模型



目录结构：

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



### Model List



| model_name | finetuned_dataset    |
| ---------- | -------------------- |
| tapas      | sqa, wikisql, wikitq |
| tapex      | sqa, wikisql, wikitq |
| reastap    | sqa, wikisql, wikitq |
| omnitab    | wikisql, wikitq      |



### Dataset List



| Dataset  |
| -------- |
| SQA      |
| wikisql  |
| wikitq   |
| hybridqa |



### Get Started



使用示例：

```python
from structuredqakit import ToolBaseForStructuredQA, BartForStructuredQA, BertForStructuredQA
from utils.dataset import WikisqlDataset, WikitablequestionsDataset, MSR_SQADataset, StructuredQADatasetFromBuilder_Tapas, StructuredQADatasetFromBuilder

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



### Metrics Result



| Dataset  | Model                                                        | Denotation Acc                                       |
| -------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| HybridQA | reastap-large                                                | 0.002                                                |
| HybridQA | reastap-large-finetuned-wtq                                  | 0.0063                                               |
|          |                                                              |                                                      |
| WikiSQL  | tapas-large-finetuned-wikisql-supervised                     | 0.7028   **(cell selection accuracy)**               |
| WikiSQL  | tapex-large-finetuned-wikisql                                | 0.8947                                               |
| WikiSQL  | **reastap-large-finetuned-wikisql**                          | **0.8956**                                           |
| WikiSQL  | [omnitab-large-finetuned-wikisql](https://huggingface.co/yilunzhao/omnitab-large-finetuned-wikisql) | 0.8874                                               |
|          |                                                              |                                                      |
| Wikitq   | tapas-large-finetuned-wtq                                    | 0.4771   **(cell selection accuracy, 1855 samples)** |
| Wikitq   | tapex-large-finetuned-wtq                                    | 0.5722                                               |
| Wikitq   | reastap-large-finetuned-wtq                                  | 0.5973                                               |
| Wikitq   | **omnitab-large-finetuned-wtq**                              | **0.6097**                                           |
|          |                                                              |                                                      |
| SQA      | tapas-large-finetuned-sqa                                    | 0.521    **(cell selection accuracy)**               |
| SQA      | tapex-large-finetuned-sqa                                    | 0.3987                                               |
| SQA      | *tapex-large-finetuned-wtq*                                  | 0.3775                                               |
| SQA      | *reastap-large-finetuned-wtq*                                | 0.3854                                               |
| SQA      | ***omnitab-large-finetuned-wtq***                            | **0.4146**                                           |



**P.S.: HybridQA 之后删掉**



### Weakness & Future work



- 目前均为 end-to-end-qa 的方式，没有生成 text-to-sql 过程
- tapas 模型需要过滤较多会出错的数据，且暂不支持聚合函数标签，只有单元格选择标签
- 由于 tapas 模型一次只能 tokenize 一个表格，故其数据预处理较慢

