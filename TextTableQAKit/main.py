'''
### Linux工程迁移
1.hybridqa.py windows不支持linux的文件命名 视为非法命名
   ```
   if platform.system() == 'Windows':
       illegal_chars = r'[\\/:\*\?"<>|]'
       table_id = re.sub(illegal_chars, '_', table_id)
   ```
'''



import math
import os
from io import BytesIO

from flask import Flask, render_template, jsonify, request, send_file, session
import pandas as pd
import random
import yaml
import json
import glob
import shutil
import logging
import sys

from TextTableQAKit.loaders import DATASET_CLASSES
from TextTableQAKit.structs.data import Table, Cell
from TextTableQAKit.utils import export

def init_app():
    flask_app = Flask(
        "TextTableQA",
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )

    flask_app.database = dict()
    flask_app.config.update(SECRET_KEY=os.urandom(24))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")) as f:
        config = yaml.safe_load(f)
    flask_app.config.update(config)
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


def statistics_default_table_information(dataset_name, split, table_idx, propertie_name_list):
    if dataset_name not in app.config['datasets']:
        raise Exception(f"datasets {dataset_name} not found")
    elif split not in app.config['split']:
        raise Exception(f"split {split} not found")
    check_data_integrity(dataset_name, split, table_idx)
    dataset_obj = app.database["dataset"][dataset_name]
    table_data = dataset_obj.get_table(split, table_idx)
    properties_html, table_html = export.export_table(table_data, export_format="html",
                                                      displayed_props=propertie_name_list)
    generated_results = fetch_generated_outputs(dataset_name, split, table_idx)
    data = {
        # "session": {},
        "table_cnt": dataset_obj.get_example_count(split),
        "generated_results": generated_results,
        "dataset_info": dataset_obj.get_info(),
        # "MultiModalQA is a dataset designed for multimodal question-answering tasks. It aims to provide a diverse range of questions that require both textual and visual understanding to answer accurately. The dataset contains questions related to images, where each question is accompanied by both text and visual information.",
        "table_question": table_data.default_question,
        "properties_html": properties_html,
        "table_html": table_html,
        "pictures": table_data.pic_info,  # 如果无，返回[]
        "text": {(index + 1): value for index, value in enumerate(table_data.txt_info)},  # 如果无，返回{}
        "success": True
    }
    print("table_cnt\n", dataset_obj.get_example_count(split))
    print("generated_results\n", generated_results)
    print("dataset_info\n", dataset_obj.get_info())
    print("table_question\n", table_data.default_question)
    with open("properties.html", "w", encoding="utf-8") as file:
        file.write(properties_html)
    with open("table.html", "w", encoding="utf-8") as file:
        file.write(table_html)
    print("pictures\n", table_data.pic_info)
    print("text\n", {(index + 1): value for index, value in enumerate(table_data.txt_info)})
    return data
# done
@app.route("/default/table", methods=["GET", "POST"])
def fetch_default_table_data():
    '''
    dataset_name = request.args.get("dataset")
    split = request.args.get("split")
    table_idx = int(request.args.get("table_idx"))
    displayed_props = json.loads(request.args.get("displayed_props"))
    '''
    # try:
        # content = request.json
        # dataset_name = content.get("dataset_name")
        # split = content.get("split")
        # table_idx = content.get("table_idx")
    dataset_name = "hybridqa"
    split = "train"
    table_idx = 20
    propertie_name_list = []
    # testing
    data = statistics_default_table_information(dataset_name, split, table_idx, propertie_name_list)

    # except Exception as e:
    #     logger.error(f"Fetch Table Error: {e}")
    #     data = {"success": False}
    return jsonify(data)


def fetch_generated_outputs(dataset_name, split, table_idx):
    # outputs = {}
    #
    # out_dir = os.path.join(app.config["root_dir"], app.config["generated_outputs_dir"], dataset_name, split)
    # if not os.path.isdir(out_dir):
    #     return outputs
    #
    # for filename in glob.glob(out_dir + "/" + "*.jsonl"):
    #     line = linecache.getline(filename, table_idx + 1)  # 1-based indexing
    #     j = json.loads(line)
    #     model_name = os.path.basename(filename).rsplit(".", 1)[0]
    #     outputs[model_name] = j
    '''
        insert your code
    '''
    outputs = {'T5-small': 'Daniel Henry Chamberlain was the 76th Governor of South Carolina in 1874.',
    'LLAMA-Lora': 'Daniel Henry Chamberlain was the 76th Governor of South Carolina in 1874.'}

    return outputs

# done
@app.route("/custom/table", methods=["GET", "POST"])
def fetch_custom_table_data():
    try:
        # json_file = request.json
        # table_name = json_file.get("table_name")
        table_name = "我的自定义表格1"
        properties_name_list = []
        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            table_data = custom_tables[table_name]
            properties_html,table_html = export.export_table(table_data, export_format="html", displayed_props=properties_name_list)
            data = {
                "table_html": table_html,
                "properties_html": properties_html,
                "success": True
            }
            with open("properties.html", "w", encoding="utf-8") as file:
                file.write(properties_html)
            with open("table.html", "w", encoding="utf-8") as file:
                file.write(table_html)
        else:
            raise Exception("fetch non-existent tables in session")
    except Exception as e:
        logger.error(f"Fetch Table Error: {e}")
        data = {"success": False}

    return jsonify(data)

# done
@app.route("/custom/remove", methods=["GET", "POST"])
def remove_custom_table():
    try:
        # table_name = request.json.get("table_name")
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

# done
@app.route("/session", methods=["GET", "POST"])
def get_session():
    try:
        # target = request.json.get("target")
        target = "all_key"
        target = "custom_tables_name"
        if target == "custom_tables_name":
            data = jsonify(list(session.get("custom_tables", {}).keys()))
        else:
            raise Exception("Illegal Target")
    except Exception as e:
        logger.error(f"Get Session Error: {e}")
        data = jsonify([])

    return data


@app.route("/pipeline", methods=["GET", "POST"])
def fetch_pipeline_result():
    pass

# done
@app.route("/custom/upload", methods=["GET", "POST"])
def upload_custom_table():
    # 上传的表格名不能重复,前端校验+后端校验 限制文件大小
    #############################################################
    # 问题四： 文件download功能
    # 问题五： pipeline功能
    try:
        # file = request.files['excel_file']
        # json_file = request.json
        # properties = json_file.get('properties')
        # table_name = json_file.get('table_name')

        file = 'test.xlsx'
        properties = {}  # 取值1
        # properties = {"title": "List of Governors of South Carolina", "overlap_subset": "True"}  # 取值2

        table_name = "我的自定义表格1"

        custom_tables = session.get("custom_tables", {})
        if table_name in custom_tables:
            raise Exception("add duplicate names to tables in session")
        else:
            df = pd.read_excel(file, dtype=str)
            headers = df.columns.tolist()
            data = df.values.tolist()
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

    for key in properties:
        t.props[key] = properties[key]

    for header_cell in headers:
        c = Cell()
        # if math.isnan(header_cell):
        #     header_cell = ""
        if isinstance(header_cell, float) and math.isnan(header_cell):
            header_cell = ""
        c.value = header_cell
        c.is_col_header = True
        t.add_cell(c)
    t.save_row()

    for row in data:
        for cell in row:
            c = Cell()
            # if math.isnan(cell):
            #     cell = ""
            if isinstance(cell, float) and math.isnan(cell):
                cell = ""
            c.value = cell
            t.add_cell(c)
        t.save_row()

    return t
def download_table(format, table_data, include_props, file_name):
    if format == "txt":
        content = export.table_to_linear(table_data, include_props)
        file_stream = BytesIO(content.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype="text/plain",
            download_name=f"{file_name}.{format}",
            as_attachment=True
        )
    elif format == "json":
        content = export.table_to_json(table_data, include_props)
        json_data = json.dumps(content)
        file_stream = BytesIO(json_data.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype='application/json',
            download_name=f"{file_name}.{format}",
            as_attachment=True
        )

    elif format == "xlsx":
        file_stream = export.table_to_excel(table_data, include_props)
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name=f"{file_name}.{format}",
            as_attachment=True,
        )

    elif format == "csv":
        content = export.table_to_csv(table_data)
        file_stream = BytesIO(content.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype="text/csv",
            download_name=f"{file_name}.{format}",
            as_attachment=True
        )
    elif format == "html":
        content = export.table_to_html(table_data, None, include_props, "export", merge=True)
        file_stream = BytesIO(content.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype="text/html",
            download_name=f"{file_name}.{format}",
            as_attachment=True
        )
    else:
        raise Exception("illegal format")
# done
@app.route("/custom/download", methods=["GET", "POST"])
def download_custom_table():
    try:

        format = "json"
        include_props = True
        table_name = "我的自定义表格1"

        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            table_data = custom_tables[table_name]
            return download_table(format, table_data, include_props, f"custom_{table_name}")
        else:
            raise Exception("download non-existent tables in session")
    except Exception as e:
        logger.error(f"Download Table Error: {e}")
        return jsonify(success=False)
# done
@app.route("/default/download", methods=["GET", "POST"])
def download_default_table():
    # content = request.json
    # format = content["format"]
    # include_props = content["include_props"]
    # dataset_name = content["dataset_name"]
    # split = content["split"]
    # table_idx = content["table_idx"]
    try:
        format = "csv"
        include_props = True
        dataset_name = "hybridqa"
        split = "dev"
        table_idx = 20

        if dataset_name not in app.config['datasets']:
            raise Exception(f"datasets {dataset_name} not found")
        elif split not in app.config['split']:
            raise Exception(f"split {split} not found")
        check_data_integrity(dataset_name, split, table_idx)
        dataset_obj = app.database["dataset"][dataset_name]
        table_data = dataset_obj.get_table(split, table_idx)
        return download_table(format, table_data, include_props, f"{dataset_name}_{split}_{table_idx}")
    except Exception as e:
        logger.error(f"Download Table Error: {e}")
        return jsonify(success=False)
# done
@app.route("/custom", methods=["GET", "POST"])
def custom_mode():
    return render_template("custom_mode.html")
# done
@app.route("/custom/example", methods=["GET", "POST"])
def download_file_example():
    return send_file(
        app.config['example_filename'],
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name="example.xlsx"
    )

# done
@app.route("/", methods=["GET", "POST"])
def index():

    try:
        dataset_name = app.config['default_dataset']
        split = "train"
        table_idx = 0
        propertie_name_list = []
        logger.info(f"Page loaded")
        data = statistics_default_table_information(dataset_name, split, table_idx, propertie_name_list)
    except Exception as e:
        logger.error(f"Fetch initial Table Error: {e}")
        data = {"success": False}


    return render_template(
        "index.html",
        table_data = data
    )

with app.app_context():
    app.database['dataset'] = {}
    # fetch_table_data()
    # dataset_obj = app.database['dataset']["wikisql"]
    # print(dataset_obj.tables)
    # session1 = {}
    # upload_custom_table()
    # fetch_custom_table_data()
    # session["custom_tables"] = {}
    # fetch_default_table_data()
    # pass
    # download_default_table()
    # upload_custom_table()


app.run()