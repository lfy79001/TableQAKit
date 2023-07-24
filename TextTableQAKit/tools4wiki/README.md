## Tools4wiki (to be decided)



以 ReasTAP 的代码为基础改写。

主要新增了 BaseTableTextQADataset 类，使用时需要继承该类，实现 get_example 方法，用于返回 (问题，表格，答案) 三元组



不足：

目前数据集加载需要用户自己实现，适配的数据集较少，且不知道对于新数据集的适配程度。

多卡训练还没试



目录结构：

```
└─tools4wiki
    └─outputs   # default directory to store the results
    └─utils
        ├─common.py
        ├─dataset.py
        ├─wikisql_utils.py
    └─tools4wiki.py
```





使用示例：

```python
from tools4wiki import Tools4Wiki
from utils.dataset import WikisqlDataset

tool = Tools4Wiki(model_name_or_path = 'path-to-pretrained-model')
dataset = WikisqlDataset()
tool.train(dataset=dataset, per_device_train_batch_size = 4)
tool.eval(dataset=dataset)
tool.predict(dataset= dataset)
```