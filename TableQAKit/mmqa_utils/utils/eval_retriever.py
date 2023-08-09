import time
import json
import argparse
import copy
import os

from typing import List
import platform
import multiprocessing

from utils import load_data_split, get_type
from utils.retriever import Retriever

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def work_eval(
        args,
        g_eids: List,
        dataset,
        retriever
):  
    g_dict = dict()
    total_gold_passages = 0
    total_gold_images = 0
    total_retrieved_passages = 0
    total_retrieved_images = 0
    gold_retrieved_passages = 0
    gold_retrieved_images = 0
    max_gold_passages = 0
    max_gold_images = 0
    gold_types = 0
    print(len(g_eids))
    for g_eid in g_eids:
        g_data_item = dataset[g_eid]
        g_dict[g_eid] = {
            'generations': dict(),
            'ori_data_item': copy.deepcopy(g_data_item)
        }
        table = g_data_item['table']
        new_passages, new_images, qtype = retriever.retrieve(g_data_item, g_eid, g_data_item['id'])
        # if qtype is None:
            # AssertionError("qtype is None")
            
        if (qtype is not None) and qtype == get_type(g_data_item['type']):
            gold_types += 1

        supc = g_data_item['supporting_context']
        gold_passage_id = [id for idx,id in enumerate(supc['doc_id']) if supc['doc_part'][idx] == 'text']
        gold_image_id   = [id for idx,id in enumerate(supc['doc_id']) if supc['doc_part'][idx] == 'image']

        gold_passage_id = list(set(gold_passage_id))
        gold_image_id = list(set(gold_image_id))
        

        if len(gold_passage_id) > max_gold_passages:
            max_gold_passages = len(gold_passage_id)
            # print(g_data_item['question'])

        if len(gold_image_id) > max_gold_images:
            max_gold_images = len(gold_image_id)
            # print(g_data_item['question'])

        if len(gold_passage_id) > 5:
            print(g_data_item['question'])
            
        total_gold_passages += len(gold_passage_id)
        total_gold_images += len(gold_image_id)
        total_retrieved_passages += len(new_passages['id'])
        total_retrieved_images += len(new_images['id'])
        gold_retrieved_passages += len(set(gold_passage_id) & set(new_passages['id']))
        gold_retrieved_images += len(set(gold_image_id) & set(new_images['id']))

    print(f"Type Accuracy:\t {gold_types / len(g_eids)}")
    print(f"Max gold passages:\t {max_gold_passages}")
    print(f"Max gold images:\t {max_gold_images}")
    print(f"Total gold passages:\t {total_gold_passages}")
    print(f"Total gold images:\t {total_gold_images}")
    print(f"Total retrieved passages:\t {total_retrieved_passages}")
    print(f"Total retrieved images:\t {total_retrieved_images}")
    print(f"Gold retrieved passages:\t {gold_retrieved_passages}")
    print(f"Gold retrieved images:\t {gold_retrieved_images}")
    # recall
    print(f"Passage recall:\t {gold_retrieved_passages / total_gold_passages}")
    print(f"Image recall:\t {gold_retrieved_images / total_gold_images}")
    # precision
    print(f"Passage precision:\t {gold_retrieved_passages / total_retrieved_passages}")
    print(f"Image precision:\t {gold_retrieved_images / total_retrieved_images}")
    # f1
    print(f"Passage f1:\t {2 * gold_retrieved_passages / (total_gold_passages + total_retrieved_passages)}")
    print(f"Image f1:\t {2 * gold_retrieved_images / (total_gold_images + total_retrieved_images)}")
    # total
    print(f"Total recall:\t {(gold_retrieved_passages + gold_retrieved_images) / (total_gold_passages + total_gold_images)}")
    print(f"Total precision:\t {(gold_retrieved_passages + gold_retrieved_images) / (total_retrieved_passages + total_retrieved_images)}")
    print(f"Total f1:\t {2 * (gold_retrieved_passages + gold_retrieved_images) / (total_gold_passages + total_gold_images + total_retrieved_passages + total_retrieved_images)}")



def main(args):
    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)
    generate_eids = list(range(len(dataset)))
    retriever = Retriever()
    work_eval(args,generate_eids,dataset, retriever)


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='mmqa',
                        choices=['mmqa'])
    parser.add_argument('--dataset_split', type=str,
                        default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--retriever', type=str,default='bd',choices=['dpmlb', 'bd'])

    # debug options
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    main(args)
