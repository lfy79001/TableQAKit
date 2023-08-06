import os
import json
from .args import RunnningArguments, get_running_args
from transformers import HfArgumentParser
import time
import json
import argparse
import copy
import os

from typing import List
import platform
import multiprocessing

from .qa import openai_qa
from .qa.openai_qa import Generator
from .utils import load_data_split, parse_ans, get_type
from utils.retriever import Retriever

class MMQA_Runner:
    def __init__(self, api_keys=None) -> None:
        self.args = get_running_args()
        # Build paths
        args = self.args
        os.makedirs(args.save_dir, exist_ok=True)

        # Load dataset
        
        self.dataset = load_data_split(dataset_to_load='mmqa',split=args.split, data_dir=args.data_path) # 从本地加载加载数据集

        # Load openai keys
        if api_keys==None and args.api_keys_file is not None:
            with open(args.api_keys_file, 'r') as f:
                self.keys = [line.strip() for line in f.readlines()]
        elif api_keys!=None:
            self.keys = api_keys
        else:
            raise ValueError('no api keys provided')
    
    def worker_qa(
        self, 
        pid: int,
        args,
        generator: Generator,
        g_eids: List,
        dataset
    ):
        """
        A worker process for end to end qa.
        """
        retriever = Retriever(args)
        g_dict = dict()
        built_few_shot_prompts = []
        with open(os.path.join(args.template_path, args.prompt_file), 'r') as f:
            build_cfg = json.load(f)

        try:
            for g_eid in g_eids:
                try:
                    g_data_item = dataset[g_eid]
                    if args.oracle_retriever:
                        new_passages = {"id": [], "title": [], "text": [], "url": []}
                        new_images = {"id": [], "title": [], "pic": [], "url": [], "path": []}
                        # Get the retrieved passages
                        for _id, title, text, url in zip(g_data_item['passages']['id'],
                                                    g_data_item['passages']['title'],
                                                    g_data_item['passages']['text'],
                                                    g_data_item['passages']['url']):
                            if _id in g_data_item['supporting_context']['doc_id']:
                                new_passages["id"].append(_id)
                                new_passages["title"].append(title)
                                new_passages["text"].append(text)
                                new_passages["url"].append(url)
                        
                        # Get the retrieved images
                        for _id, title, pic, url, path in zip(g_data_item['images']['id'],
                                                        g_data_item['images']['title'],
                                                        g_data_item['images']['pic'],
                                                        g_data_item['images']['url'],
                                                        g_data_item['images']['path']):
                            if _id in g_data_item['supporting_context']['doc_id']:
                                new_images["id"].append(_id)
                                new_images["title"].append(title)
                                new_images["pic"].append(pic)
                                new_images["url"].append(url)
                                new_images["path"].append(path)
                    else:
                        new_passages, new_images, qtype = retriever.retrieve(g_data_item, g_eid, g_data_item['id'])
                    if args.oracle_classifier:
                        qtype = get_type(g_data_item['type'])

                    g_data_item['passages'] = new_passages
                    g_data_item['images'] = new_images
                    g_dict[g_eid] = {
                        'generations': '',
                        'ori_data_item': copy.deepcopy(g_data_item)
                    }

                    prompt_cfg = build_cfg[qtype].copy()
                    
                    # Build prompt
                    generate_prompt = generator.build_generate_prompt(
                        data_item=g_data_item,
                        qtype=qtype,
                        cot=prompt_cfg['cot']
                    )

                    ok = False
                    n_shots = prompt_cfg['n-shot']
                    while not ok:

                        built_few_shot_prompts = []
                        few_shot_prompt = generator.build_few_shot_prompt_from_file(
                            file_path=prompt_cfg['file'],
                            n_shots=n_shots,
                            qtype = qtype
                        )
                        
                        prompt = few_shot_prompt + "\n\n\n" + generate_prompt # 手造的 few shot prompt + 需要提交的 prompt

                        print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
                        
                        built_few_shot_prompts.append((g_eid, prompt))

                        print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run {args.engine} API.")

                        ok, response_dict = generator.generate_one_pass(
                            prompts=built_few_shot_prompts,
                            args = prompt_cfg, 
                            verbose=args.verbose
                        )

                        if not ok:
                            assert n_shots >= 0, "n_shots should be no less than 0."
                            n_shots -= 1
                    for eid, g_pairs in response_dict.items():
                        g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                        pred_ans = g_pairs[0][0]

                        for g_pair in g_pairs:
                            status, ans_text = parse_ans(g_pair[0],qtype, cot=prompt_cfg['cot'])
                            if status:
                                pred_ans = ans_text
                                break
                        g_dict[eid]['generations'] = pred_ans.strip()
                    built_few_shot_prompts = []
                except Exception as e:
                    print(f"Process#{pid}: eid#{g_eid}, qid#{g_data_item['id']} generation error: {e}")
                    break
        except KeyboardInterrupt:
            g_dict.popitem()
            print(f"Process#{pid}: Keyboard interrupt. Saving the generations.")
        return g_dict

    def predict(self):
        start_time = time.time()
        args = self.args
        dataset = self.dataset
        keys = self.keys
        # End To End QA
        generator = openai_qa.Generator(args, keys) # 调用 LLM api 的
        generate_eids = list(range(len(dataset)))
        generate_eids_group = [[] for _ in range(args.n_processes)]
        for g_eid in generate_eids:
            generate_eids_group[int(g_eid) % args.n_processes].append(g_eid) # 每个进程处理的数据集, 按剩余系分
        print('\n******* End To End QA *******')
        g_dict = dict()
        worker_results = []
        if args.n_processes > 1:
            pool = multiprocessing.Pool(processes=args.n_processes)
            for pid in range(args.n_processes): # 每个 api_key 对应一个进程
                worker_results.append(pool.apply_async(self.worker_qa, args=(
                    pid,
                    args,
                    generator,
                    generate_eids_group[pid],
                    dataset,
                )))

            # Merge results
            for r in worker_results:
                worker_g_dict = r.get()
                g_dict.update(worker_g_dict)
            pool.close()
            pool.join()
            # Save results
            # "_".join(["{}={}".format(k, str(args.__dict__[k])) for k in args.__dict__ if k not in ['api_keys_file', 'prompt_file', 'save_dir', 'stop_tokens']])
            save_file_name = f'mmhqa-icl_mmqa_{args.split}.json'
            with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
                json.dump(g_dict, f, indent=4)
        else:
            """
            for resume qa with n_processes=1
            """
            if args.save_file_name is None:
                save_file_name = f'mmhqa-icl_mmqa_{args.split}.json'
            else :
                save_file_name = args.save_file_name

            begin_eid = 0
            end_eid = len(generate_eids)

            g_dict = dict()
            if args.resume:
                try:
                    with open(os.path.join(args.save_dir, save_file_name), 'r') as f:
                        g_dict = json.load(f)
                        begin_eid = len(g_dict)
                except Exception as e:
                    print(e)
                    print(f"Resume failed")
                    exit(0)

            print("begin end to end qa begin_eid: ", begin_eid, "end_eid: ", end_eid, "total: ", len(generate_eids)) 

            n_g_dict = self.worker_qa(
                0,
                args,
                generator,
                generate_eids[begin_eid:end_eid], # 从上次处理的位置开始
                dataset,
            )
            g_dict.update(n_g_dict)

            with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
                json.dump(g_dict, f, indent=4)

        print(f"Elapsed time: {time.time() - start_time}")