import math
import os
from io import BytesIO

from flask import Flask, render_template, jsonify, request, send_file, session
from flask_cors import CORS
import pandas as pd
import random
import yaml
import json
import glob
import shutil
import logging
import sys
import traceback
import pickle
from loaders import DATASET_CLASSES
from structs.data import Table, Cell
from utils import export
from try_table_qa_kit import TableQAKitDemo

def init_app():
    flask_app = Flask(
            "TextTableQA",
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static")
        )

    flask_app.database = {}
    flask_app.database['dataset'] = {}
    flask_app.config.update(SECRET_KEY=os.urandom(24))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")) as f:
        config = yaml.safe_load(f)
    flask_app.config.update(config)
    flask_app.pipeline = TableQAKitDemo(model_type="text_davinci_003", key="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")

    
    return flask_app

app = init_app()
CORS(app)


def init_log():
    log_format = "%(levelname)s:"
    logging.basicConfig(format=log_format, level=logging.INFO)
    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(__name__)
    return logger
logger = init_log()





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
    properties_html, table_html = export.table_to_html(table_data, propertie_name_list,None,"web")
    generated_results = fetch_generated_outputs(dataset_name, split, table_idx)
    data = {
        # "session": {},
        "table_cnt": dataset_obj.get_example_count(split),
        "generated_results": generated_results,
        "dataset_info": dataset_obj.get_info(),
        "table_question": table_data.default_question,
        "properties_html": properties_html,
        "table_html": table_html,
        "pictures": table_data.pic_info,  # 如果无，返回[]
        "text": {(index + 1): value for index, value in enumerate(table_data.txt_info)},  # 如果无，返回{}
        "success": True
    }
    # print("table_cnt\n", dataset_obj.get_example_count(split))
    # print("generated_results\n", generated_results)
    # print("dataset_info\n", dataset_obj.get_info())
    # print("table_question\n", table_data.default_question)
    # with open("properties.html", "w", encoding="utf-8") as file:
    #     file.write(properties_html)
    # with open("table.html", "w", encoding="utf-8") as file:
    #     file.write(table_html)
    # print("pictures\n", table_data.pic_info)
    # print("text\n", {(index + 1): value for index, value in enumerate(table_data.txt_info)})
    return data
# done
@app.route("/default/table", methods=["GET", "POST"])
def fetch_default_table_data():
    try:
        content = request.json
        dataset_name = content.get("dataset_name")
        split = content.get("split")
        table_idx = content.get("table_idx")
    # dataset_name = "MultiHiertt"
    # split = "train"
    # table_idx = 0
        propertie_name_list = []
        data = statistics_default_table_information(dataset_name, split, table_idx, propertie_name_list)
        return jsonify(data),200

    except Exception as e:
        logger.error(f"Fetch Table Error: {e}")
        data = {"success": False}
        return jsonify(data),400 


def fetch_generated_outputs(dataset_name, split, table_idx):
    
    outputs = {}

    model_list = os.listdir(f"outputs/{dataset_name}")
    valid_model_list = []
    for model_name in model_list:
        split_list = os.listdir(f"outputs/{dataset_name}/{model_name}")
        for split_file_name in split_list:
            split_name, _ = os.path.splitext(split_file_name)
            if split == split_name:
                valid_model_list.append(model_name)
                break
    for valid_model_name in valid_model_list:
        outputs_file_path = f'outputs/{dataset_name}/{valid_model_name}/{split}.txt'
        with open(outputs_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            outputs[valid_model_name] = lines[table_idx].strip()        

    return outputs

# done
@app.route("/custom/table", methods=["GET", "POST"])
def fetch_custom_table_data():
    try:
        json_file = request.json
        table_name = json_file.get("table_name")
        # table_name = "我的自定义表格1"
        properties_name_list = []
        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            table_data = pickle.loads(custom_tables[table_name])
            properties_html,table_html = export.table_to_html(table_data, properties_name_list,None,"web")
            data = {
                "table_html": table_html,
                "properties_html": properties_html,
                "success": True
            }
            # with open("properties.html", "w", encoding="utf-8") as file:
            #     file.write(properties_html)
            # with open("table.html", "w", encoding="utf-8") as file:
            #     file.write(table_html)
            return jsonify(data),200
        else:
            raise Exception("fetch non-existent tables in session")
    except Exception as e:
        logger.error(f"Fetch Table Error: {e}")
        data = {"success": False}
        return jsonify(data),400

# done
@app.route("/custom/remove", methods=["GET", "POST"])
def remove_custom_table():
    try:
        json_file = request.json
        table_name = json_file.get("table_name")
        # table_name = "我的自定义表格1"
        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            custom_tables.pop(table_name)
        else:
            raise Exception("delete non-existent tables in session")
        session["custom_tables"] = custom_tables
        session.modified = True
        result = {"success" : True}
        return jsonify(result),200
    except Exception as e:
        logger.error(f"Remove Table Error: {e}")
        result = {"success" : False}
        return jsonify(result),400

# done
@app.route("/session", methods=["GET", "POST"])
def get_session():
    try:
        json_file = request.json
        target = json_file.get("target")
        # target = "custom_tables_name"
        if target == "custom_tables_name":
            data = {
                "data" : list(session.get("custom_tables", {}).keys()), #["a","b"]
                "success" : True
            }
            return jsonify(data),200
        else:
            raise Exception("Illegal Target")
    except Exception as e:
        logger.error(f"Get Session Error: {e}")
        data = {"success" : False}
        return jsonify(data),400
    


@app.route("/default/pipeline", methods=["GET", "POST"])
def fetch_default_pipeline_result():
    try:
        content = request.json
        dataset_name = content.get("dataset_name")
        split = content.get("split")
        table_idx = content.get("table_idx")
        question = content.get("question")


        if dataset_name not in app.config['datasets']:
            raise Exception(f"datasets {dataset_name} not found")
        elif split not in app.config['split']:
            raise Exception(f"split {split} not found")

        answer = app.pipeline.try_table_qa_kit_by_id(data=dataset_name, dataset_type=split, index=table_idx, question=question)
        data = {
            "answer": answer,
            "success": True
        }
        return jsonify(data),200
        
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        data = {"success": False}
        return jsonify(data),400

@app.route("/custom/pipeline", methods=["GET", "POST"])
def fetch_custom_pipeline_result():
    try:
        content = request.json
        table_name = content.get("table_name")
        question = content.get("question")

        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            table_data = pickle.loads(custom_tables[table_name])
            cells = table_data.cells
            input = [[cell.value for cell in row] for row in cells]
            answer = app.pipeline.try_table_qa_kit(table=input, texts=None, question=question)

            data = {
                "answer": answer,
                "success": True
            }

            return jsonify(data),200
        else:
            raise Exception("fetch non-existent tables in session")
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        data = {"success": False}
        return jsonify(data),400


# done
@app.route("/custom/upload", methods=["GET", "POST"])
def upload_custom_table():
    # 上传的表格名不能重复,前端校验+后端校验 限制文件大小
    #############################################################
    try:
        if 'excel_file' in request.files:
            file = request.files['excel_file']
        else:
            raise Exception("no excel file")
        # json_file = request.json
        # table_name = json_file.get('table_name')
    
        table_name, file_extension = os.path.splitext(file.filename)

        # file = 'test.xlsx'
        properties = {}
        # properties = {"title": "List of Governors of South Carolina", "overlap_subset": "True"}

        # table_name = "我的自定义表格1"

        custom_tables = session.get("custom_tables", {})
        if table_name in custom_tables:
            raise Exception("add duplicate names to tables in session")
        else:     
            if file_extension == '.xls':
                df = pd.read_excel(file, dtype=str, engine="xlrd")
            elif file_extension == '.xlsx':
                df = pd.read_excel(file, dtype=str, engine="openpyxl")
            else:
                raise Exception("Unsupported file format. Only .xls and .xlsx are supported.")

            headers = df.columns.tolist()
            data = df.values.tolist()
            table_data = prepare_custom_table(headers, data, properties, table_name)
            custom_tables[table_name] = pickle.dumps(table_data)
        session["custom_tables"] = custom_tables
        session.modified = True
        result = {"success" : True}
        return jsonify(result),200
    except Exception as e:
        logger.error(f"Upload Table Error: {e}")
        result = {"success" : False}
        return jsonify(result),400

    


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
        ),200
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
        ),200

    elif format == "xlsx":
        file_stream = export.table_to_excel(table_data, include_props)
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name=f"{file_name}.{format}",
            as_attachment=True,
        ),200

    elif format == "csv":
        content = export.table_to_csv(table_data)
        file_stream = BytesIO(content.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype="text/csv",
            download_name=f"{file_name}.{format}",
            as_attachment=True
        ),200
    elif format == "html":
        content = export.table_to_html(table_data, None, include_props, "export", merge=True)
        file_stream = BytesIO(content.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype="text/html",
            download_name=f"{file_name}.{format}",
            as_attachment=True
        ),200
    else:
        raise Exception("illegal format")
# done
@app.route("/custom/download", methods=["GET", "POST"])
def download_custom_table():
    try:
        format = request.args.get('format')
        table_name = request.args.get('table_name')
        # format = "json"
        include_props = False
        # table_name = "我的自定义表格1"

        custom_tables = session.get("custom_tables", {})
        if len(custom_tables) != 0 and table_name in custom_tables:
            table_data = pickle.loads(custom_tables[table_name])
            return download_table(format, table_data, include_props, f"custom_{table_name}")
        else:
            raise Exception("download non-existent tables in session")
    except Exception as e:
        logger.error(f"Download Table Error: {e}")
        return jsonify({"success" : False}),400
# done
@app.route("/default/download", methods=["GET", "POST"])
def download_default_table():
    try:
        format = request.args.get('format')
        include_props = request.args.get('include_props')
        dataset_name = request.args.get('dataset_name')
        split = request.args.get('split')
        table_idx = int(request.args.get('table_idx'))


        if dataset_name not in app.config['datasets']:
            raise Exception(f"datasets {dataset_name} not found")
        elif split not in app.config['split']:
            raise Exception(f"split {split} not found")
        elif dataset_name == "MultiHiertt" and format != "html":
            raise Exception(f"dataset {dataset_name} can only be downloaded in HTML format")
        check_data_integrity(dataset_name, split, table_idx)
        dataset_obj = app.database["dataset"][dataset_name]
        table_data = dataset_obj.get_table(split, table_idx)
        return download_table(format, table_data, include_props, f"{dataset_name}_{split}_{table_idx}")
    except Exception as e:
        logger.error(f"Download Table Error: {e}")
        logger.error(e.__traceback__)
        return jsonify({"success" : False}),400
# done
@app.route("/custom", methods=["GET", "POST"])
def custom_mode():
    return render_template("custom_mode.html")
# done
@app.route("/custom/example", methods=["GET", "POST"])
def download_file_example():
    try:
        file_response = send_file(
            app.config['example_filename'],
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name="example.xlsx"
        ),200
        return file_response
    except Exception as e:
        logger.error(f"Download Example Error: {e}")
        return jsonify({"success" : False}),400

# done
@app.route("/", methods=["GET", "POST"])
def index():

    # try:
    #     dataset_name = app.config['default_dataset']
    #     split = "train"
    #     table_idx = 0
    #     propertie_name_list = []
    #     logger.info(f"Page loaded")
    #     data = statistics_default_table_information(dataset_name, split, table_idx, propertie_name_list)
    # except Exception as e:
    #     logger.error(f"Fetch initial Table Error: {e}")
    #     data = {"success": False}


    return render_template(
        "index.html"
        # table_data = data  #--这样传没有jsonfiy应该是可以的--
    )

with app.app_context():
    print("Initing datasets")
    for dataset_name in app.config['datasets']:
        for split in app.config['split']:
            check_data_integrity(dataset_name, split, 0)
    print("done")
#     import psutil
#     # 获取内存使用情况
#     memory_info = psutil.virtual_memory()

#     # 打印内存总量、使用量、可用量等信息
#     print(f"1Total Memory: {memory_info.total} bytes")
#     print(f"1Used Memory: {memory_info.used} bytes")
#     print(f"1Available Memory: {memory_info.available} bytes")
#     print(f"1Percentage Used: {memory_info.percent}%")
#     print("load start")
#     for dataset_name in app.config['datasets']:
#         for split in app.config['split']:
#             check_data_integrity(dataset_name, split, 0)
#     print("load over")
#     # 获取内存使用情况
#     memory_info = psutil.virtual_memory()

#     # 打印内存总量、使用量、可用量等信息
#     print(f"2Total Memory: {memory_info.total} bytes")
#     print(f"2Used Memory: {memory_info.used} bytes")
#     print(f"2Available Memory: {memory_info.available} bytes")
#     print(f"2Percentage Used: {memory_info.percent}%")



    # print(fetch_generated_outputs("spreadsheetqa","dev",10))



#     fetch_table_data()
#     dataset_obj = app.database['dataset']["wikisql"]
#     print(dataset_obj.tables)
#     session1 = {}
#     upload_custom_table()
#     fetch_custom_table_data()
#     session["custom_tables"] = {}
#     fetch_default_table_data()
#     pass
#     download_default_table()
#     upload_custom_table()
# if __name__ == '__main__':
#     app.run(host = "210.75.240.136", port = 18888)
