import datasets
import os
import json


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_data(path_dict):
    if 'train' in path_dict:
        train_data = load_json(path_dict['train'])
    if 'dev' in path_dict:
        dev_data = load_json(path_dict['dev'])
    if 'test' in path_dict:
        test_data = load_json(path_dict['test'])
    return {'train':train_data, 'dev':dev_data, 'test':test_data}



def dataset_download(dataset_name):
    root_path = '../cache/dataset'
    if dataset_name == 'hybridqa':
        dataset_path = os.path.join(root_path, dataset_name)
        if (not os.path.exists(dataset_path)) or len(os.listdir(dataset_path)) == 0:
            # 需要写下载数据集的代码
            dataset_path = '/home/lfy/beiyongdata/HybridQA'
            pass
        else:
            pass
        train_path = os.path.join(dataset_path, "train.p.json")
        dev_path = os.path.join(dataset_path, "dev.p.json")
        test_path = os.path.join(dataset_path, "test.p.json")
    elif dataset_name == 'finqa':
        pass    
    return {"train": train_path, "dev": dev_path, "test": test_path}


import logging
import sys
import math

def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log




    
    


        