# TextTableQAKit
A toolkit for Text-Table Hybrid Question Answering


# 目前思路
1. 统一数据数据模板
2. 经典模型各个模块的多种实现
3. 最新LLM的融合（检索能力，工具使用能力）
4. 模块化设计便于下游用户针对性的定制


## QuickStart
```
pip install ttqakit
ttqakit run --host=127.0.0.1 --port 13000
# 如果以上命令是在服务器上运行的，想在本地看网页，需要监听
ssh -L 8000:127.0.0.1:13000 lfy@210.75.240.136
# 在本地打开  http://127.0.0.1:8000
```

## Datasets
| Dataset                                                                              | Source                                                                                                                                          | Data type      | # train | # dev  | # test | License     |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------- | ------ | ------ | ----------- |
| **[TAT-QA](https://huggingface.co/datasets/kasnerz/cacapo)**                         | [van der Lee et al. (2020)](https://aclanthology.org/2020.inlg-1.10.pdf)                                                                        | Key-value      | 15,290  | 1,831  | 3,028  | CC BY       |
| **[FinQA](https://huggingface.co/datasets/GEM/dart)**                                 | [Nan et al. (2021)](https://aclanthology.org/2021.naacl-main.37/)                                                                               | Graph          | 62,659  | 2,768  | 5,097  | MIT         |
| **[Multihiertt](https://huggingface.co/datasets/GEM/dart)**                                 | [Nan et al. (2021)](https://aclanthology.org/2021.naacl-main.37/)                                                                               | Graph          | 62,659  | 2,768  | 5,097  | MIT         |
| **[HiTab](https://huggingface.co/datasets/kasnerz/hitab)**                           | [Cheng et al. (2021)](https://aclanthology.org/2022.acl-long.78/)                                                                               | Table          | 7,417   | 1,671  | 1,584  | C-UDA       |
| **[WikiSQL](https://huggingface.co/datasets/wikisql)**                               | [Zhong et al. (2017)](https://arxiv.org/abs/1709.00103)                                                                                         | Table + SQL    | 56,355  | 8,421  | 15,878 | BSD         |
| **[WTQuestions](https://huggingface.co/datasets/wikisql)**                               | [Zhong et al. (2017)](https://arxiv.org/abs/1709.00103)                                                                                         | Table + SQL    | 56,355  | 8,421  | 15,878 | BSD         |
| **[MultimodalQA](https://huggingface.co/datasets/wikisql)**                               | [Zhong et al. (2017)](https://arxiv.org/abs/1709.00103)                                                                                         | Table + SQL    | 56,355  | 8,421  | 15,878 | BSD         |
| **[MultimodalQA](https://huggingface.co/datasets/wikisql)**                               | [Zhong et al. (2017)](https://arxiv.org/abs/1709.00103)                                                                                         | Table + SQL    | 56,355  | 8,421  | 15,878 | BSD         |




## 如何往PyPI上提交
0. 安装必要的工具
   ```bash
   pip install setuptools wheel twine
1. 先修改setup.py这个文件
2. 生成分发文件。在命令行中运行以下命令以生成源代码压缩包和轮子（wheel）分发文件：
   ```bash
    python setup.py sdist bdist_wheel
3. 使用twine上传到测试 PyPI。运行以下命令以将您的分发文件上传到测试 PyPI：
    ```bash 
    twine upload --repository-url https://test.pypi.org/legacy/dist/*
    ```
    UserName: lfy79001

    PassWord: 20010213lfyLFY!
4. 访问 https://test.pypi.org/project/ttqakit
5. 上传到正式的
    ```bash
    twine upload dist/*
    ```
    访问 https://pypi.org/project/ttqakit
6. 完毕后可以安装
    ```bash
    # 正式版
    pip install ttqakit
    # test版
    pip install --index-url https://test.pypi.org/simple/ttqakit





