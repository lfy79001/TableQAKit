# TextTableQAKit
A toolkit for Text-Table Hybrid Question Answering


# 目前思路
1. 统一数据数据模板
2. 经典模型各个模块的多种实现
3. 最新LLM的融合（检索能力，工具使用能力）
4. 模块化设计便于下游用户针对性的定制

# 如何往PyPI上提交
0. 安装必要的工具
   ```bash
   pip install setuptools wheel twine
1. 先修改setup.py这个文件
2. 生成分发文件。在命令行中运行以下命令以生成源代码压缩包和轮子（wheel）分发文件：
   ```bash
    python setup.py sdist bdist_wheel
3. 使用twine上传到测试 PyPI。运行以下命令以将您的分发文件上传到测试 PyPI：
    ```bash 
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    ```
    UserName: lfy79001

    PassWord: 20010213lfyLFY!
4. 访问https://test.pypi.org/project/ttqakit看能不能找到
5. 上传到正式的
    ```bash
    twine upload dist/*
    ```
    访问https://pypi.org/project/ttqakit
6. 完毕后可以安装
    ```bash
    # 正式版
    pip install ttqakit
    # test版
    pip install --index-url https://test.pypi.org/simple/ ttqakit





