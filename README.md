<div align="center">
  <img src="figs/tableqakit.png" border="0" width=450px/>
  <br />
  <br />


[🌐Website](https://www.baidu.com/) |
[📦PyPI](https://www.baidu.com/)

<!-- [📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started.html#installation) | -->


</div>

# TableQAKit: A Comprehensive and Practical Toolkit for Table-based Question Answering

# 🔥 Updates

- [**2023-8-7**]: We released our [code](https://github.com/lfy79001/TableQAKit) and [PyPI](https://www.baidu.com). Check it out!

# ✨ Features
TableQAKit is a unified platform for TableQA (especially in the LLM era). Its main features includes:
- **Extensible disign**: You can use the interfaces defined by the toolkit, extend methods and models, and implement your own new models based on your own data.
- **Equipped with LLM**: TableQAKit supports LLM-based methods, including LLM-prompting methods and LLM-finetuning methods.
- **Comprehensive datasets**: We design a unified data interface to process data and store them in Huggingface datasets.
- **Powerful methods**: Using our toolkit, you can reproduce most of the SOTA methods for TableQA tasks.
- **Efficient LLM benchmark**: TableQAEval, a benchmark to evaluate the performance of LLM for TableQA. It evaluates LLM's modeling ability of long tables (context) and comprehension capabilities (numerical reasoning, multi-hop reasoning).
- **Comprehensive Survey**: We are about to release a systematic TableQA Survey, this project is a pre-work.

# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview




# 🗃️ Dataset
<p align="center">
<img src="figs/dataset_examples.png" width="512">
</p>


<p align="center">
<img src="figs/table.png" width="512">
</p>



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
pip install gunicorn

# 运行在210.75.240.136:18889,访问：http://210.75.240.136:18889
gunicorn -c gunicorn_config.py app:app --daemon

# 想要停止运行？
在gunicorn_error.log找到最新的记录Listening记录，如"Listening at: http://210.75.240.136:18889 (2609966)"
使用 kill 2609966 可实现停止运行

```






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

## HybridQA数据处理
首先需要从google drive上下载这个数据

https://drive.google.com/file/d/1MGfxoOIyoUVQEBnFXWf_jVfFiXMifXbu/view?usp=share_link

将数据集解压后放在 TextTableQAKit/modules/ 中，但是挺大的，之后git push的时候数据集得删掉，放到你本地就行
数据包括 train.json, dev.json, test.json, 还有一个文件夹包括table和passage的实际信息。

这个数据集和multimodalQA不一样，文本是呈现一个“链接”的形式，即文本不是附着在表格旁边，而是可以点击这个表格cell，弹出链接文本（这个感觉较复杂，不用实现），现阶段，就把文本附着在下面就可以。

检索代码 TextTableQAKit/modules/retrieve_hybridqa.py

python retrieve_hybridqa.py 可以直接运行.（没下载BERT所以只能运行dataset部分）

不需要运行全部的代码，主要的数据处理在Dataset的__init__()里面，在里面打断点，看看数据怎么处理的。

train/dev/test 文件里只包含了Table的id，所以需要根据这个id找到这个table对应的json文件。

Table的Json文件里，存储了表格的header和cell

cell的格式是   [ [1], [2]] ，位置1是cell的直接文本，位置2是cell链接的passage的链接，所以目前只需要先把1用好就行。








