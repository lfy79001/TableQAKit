import os
from flask import Flask, render_template, jsonify, request, send_file, session
import pandas as pd
import random
import yaml
import json
import glob
import shutil
import logging

from TextTableQAKit.loaders import DATASET_CLASSES
from TextTableQAKit.structs.data import Table, Cell
from TextTableQAKit.utils.export import export_table


def init_app():
    flask_app = Flask(
        "TextTableQA",
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )

    flask_app.database = dict()
    flask_app.config.update(SECRET_KEY=os.urandom(24))
    return flask_app


def init_logging():
    log_format = "%(levelname)s:"
    logging.basicConfig(format=log_format, level=logging.INFO)
    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(__name__)
    return logger


app = init_app()
logger = init_logging()

'''
check data integrity
'''


def check_data_integrity(dataset_name, split, table_idx):
    check_dataset_in_database(dataset_name)
    check_data_in_dataset(dataset_name, split)
    check_table_in_dataset(dataset_name, split, table_idx)


def check_dataset_in_database(dataset_name):
    is_exist = dataset_name in app.database["dataset"]
    if not is_exist:
        logger.info(f"Creating Dataset Object {dataset_name}")
        create_dataset(dataset_name)


def create_dataset(dataset_name):
    dataset_obj = DATASET_CLASSES[dataset_name]()
    app.database['dataset'][dataset_name] = dataset_obj


def check_data_in_dataset(dataset_name, split):
    dataset_obj = app.database['dataset'][dataset_name]
    if not dataset_obj.has_split(split):
        logger.info(f"Loading {dataset_name} - {split}")
        dataset_obj.load(split=split)

def check_table_in_dataset(dataset_name, split, table_idx):
    dataset_obj = app.database['dataset'][dataset_name]
    is_exist = table_idx in dataset_obj.tables[split]
    if not is_exist:
        entry = dataset_obj.get_data(split, table_idx)
        table = dataset_obj.prepare_table(entry)
        dataset_obj.set_table(split, table_idx, table)

@app.route("/data", methods=["GET", "POST"])
def fetch_table_data():
    '''
    dataset_name = request.args.get("dataset")
    split = request.args.get("split")
    table_idx = int(request.args.get("table_idx"))
    displayed_props = json.loads(request.args.get("displayed_props"))
    '''
    dataset_name = "wikisql"
    split = "dev"
    table_idx = 1
    ################################################################
    # 问题一： 此处propertie_name_list取值要什么格式，如果为空应该什么格式，才能在<html页面生成时>正确输出
    # 问题三： 此模块数据集还未成功下载，因此还没有做全面测试
    # 问题四： 在此处需要添加自定义表格的拉取
    propertie_name_list = None
    try:
        check_data_integrity(dataset_name, split, table_idx)
        dataset_obj = app.database["dataset"][dataset_name]
        table_data = dataset_obj.get_table(split, table_idx)
        table_html = export_table(table_data, export_format="html", displayed_props=propertie_name_list)
        data =  {
            "table_content": table_html,
            "table_cnt": dataset_obj.get_example_count(split),
            "dataset_info": dataset_obj.get_info(),
            "session": {}
        }
    except Exception as e:
        logger.error(f"Fetch Table Error: {e}")
        data = {}
    return jsonify(data)


@app.route("/custom/remove", methods=["GET", "POST"])
def remove_custom_table():
    try:
        table_name = "我的自定义表格1"
        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            custom_tables.pop(table_name)
        else:
            raise Exception("delete non-existent tables in session")
        session["custom_tables"] = custom_tables
        session.modified = True
        result = jsonify(success=True)
    except Exception as e:
        logger.error(f"Remove Table Error: {e}")
        result = jsonify(success=False)

    return result

@app.route("/custom/upload", methods=["GET", "POST"])
def upload_custom_table():
    # 上传的表格名不能重复,前端校验+后端校验
    #############################################################
    # 问题二： 对于自定义表格上传，我将空白值直接设置为nan，对不对，如果错了可能会在pipeline和html页面生成时出错，要看官方数据集处理时设置为啥
    try:
        file = 'test.xlsx'
        properties = None  # 取值1
        properties = {"title": "List of Governors of South Carolina", "overlap_subset": "True"}  # 取值2

        table_name = "我的自定义表格1"

        df = pd.read_excel(file)
        headers = df.columns.tolist()
        data = df.values.tolist()


        custom_tables = session.get("custom_tables", {})
        if table_name in custom_tables:
            raise Exception("add duplicate names to tables in session")
        else:
            table_data = prepare_custom_table(headers, data, properties, table_name)
            custom_tables[table_name] = table_data
        session["custom_tables"] = custom_tables
        session.modified = True
        result = jsonify(success=True)

    except Exception as e:
        logger.error(f"Upload Table Error: {e}")
        result = jsonify(success=False)

    return result

def prepare_custom_table(headers, data, properties, table_name):
    t = Table()
    t.type = "custom"
    t.custom_table_name = table_name
    if properties is not None:
        for key in properties:
            t.props[key] = properties[key]

    for header_cell in headers:
        c = Cell()
        c.value = header_cell
        c.is_col_header = True
        t.add_cell(c)
    t.save_row()

    for row in data:
        for cell in row:
            c = Cell()
            c.value = cell
            t.add_cell(c)
        t.save_row()

    return t

with app.app_context():
    # app.database['dataset'] = {}
    # fetch_table_data()
    # dataset_obj = app.database['dataset']["wikisql"]
    # print(dataset_obj.tables)
    # session["custom_tables"] = {}
    pass
