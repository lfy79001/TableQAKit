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
        entry = dataset_obj.getData(split, table_idx)
        table = dataset_obj.prepare_table(entry)
        dataset_obj.setTables(split, table_idx, table)

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
    propertie_name_list = ["reference"]
    try:

        check_data_integrity(dataset_name, split, table_idx)

        # dataset = get_dataset(dataset_name=dataset_name, split=split)
        # table = dataset.get_table(split=split, table_idx=table_idx)
        # html = dataset.export_table(table=table, export_format="html", displayed_props=displayed_props)
        # generated_outputs = get_generated_outputs(dataset_name=dataset_name, split=split, output_idx=table_idx)
        # dataset_info = dataset.get_info()
        #
        # data =  {
        #     "html": html,
        #     "total_examples": dataset.get_example_count(split),
        #     "dataset_info": dataset_info,
        #     "generated_outputs": generated_outputs,
        #     "session": get_session(),
        # }
        data = {}
    except Exception as e:
        logger.error(f"Fetch Table Error: {e}")
        data = {}

    return jsonify(data)


with app.app_context():
    app.database['dataset'] = {}
    fetch_table_data()
    dataset_obj = app.database['dataset']["wikisql"]
    print(dataset_obj.tables)
