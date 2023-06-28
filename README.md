# TextTableQAKit
A toolkit for Text-Table Hybrid Question Answering


# 目前思路
1. 统一数据数据模板
2. 经典模型各个模块的多种实现
3. 最新LLM的融合（检索能力，工具使用能力）
4. 模块化设计便于下游用户针对性的定制

## Flask框架简介
- cli.py：用于定义 Flask 应用程序的命令行接口（CLI）。在 cli.py 文件中，你可以使用 Flask 的 click 库定义命令行命令和参数，并将它们与 Flask 应用程序的功能关联起来。例如，你可以定义一个命令行命令，用于初始化 Flask 应用程序的数据库或执行其他任务。cli.py 文件通常与 Flask 应用程序的工具集集成在一起，可以方便地使用命令行来管理和维护应用程序。
- main.py：用于定义 Flask 应用程序的主要入口点。在 main.py 文件中，你可以定义 Flask 应用程序的路由和视图函数，并启动应用程序的服务器。这个文件通常是 Flask 应用程序的主要代码文件，用于实现应用程序的核心功能。
- config.yaml：用于定义 Flask 应用程序的配置选项。在 config.yaml 文件中，你可以指定 Flask 应用程序的各种配置选项，如数据库连接、调试模式、密钥等。这个文件通常是 Flask 应用程序的配置文件，可以方便地修改和管理应用程序的配置选项。
- static/：用于存储静态文件，如 CSS、JavaScript、图像等。Flask 应用程序会自动查找 static/ 文件夹，并将其中的静态文件提供给客户端。
- templates/：用于存储模板文件，如 HTML 文件、Jinja2 模板等。Flask 应用程序会自动查找 templates/ 文件夹，并使用其中的模板文件进行渲染。


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





