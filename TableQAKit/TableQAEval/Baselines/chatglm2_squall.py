import json, os
import openai
import argparse
import pandas as pd
import tiktoken
import time
import sys
from transformers import AutoTokenizer, AutoModel
import torch
sys.path.append('./')
sys.path.append('../')
sys.path.append('../Evaluation')

from Evaluation.emf1 import compute_emf1

from accelerate import load_checkpoint_and_dispatch


def generate_sys_prompt(args):
    if args.task == 'qa':
        return 'Given a table and a question, you need to follow a similar format to answer the question according to the table. Make sure the response is end with The answer is _ .'
    elif args.task == 'sql':
        return 'You are an SQL executor, you need to execute SQL based on the given table and SQL statement to obtain the answer. Make sure the response is end with The execution result is _'


def num_tokens_from_string(table, tokenizer):
    return len(tokenizer.tokenize(table))


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    
    model = model.eval()
    
    

    header = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    )
    sys_prompt = generate_sys_prompt(args)
    
    with open(args.file_name, 'r') as file:
        json_data = json.load(file)    
    
    if args.mode == 'toy':
        json_data = json_data[:10]
    elif args.mode == 'baby':
        json_data = json_data[100:1000]

    preds = []
    golds = []
    
    for i, data in enumerate(json_data):
        
        if args.format == 'markdown':
            df = pd.DataFrame(data['contents'], columns=data['header'])
            table_str = df.to_markdown(headers=data['header'], tablefmt="pipe")
        elif args.format == 'flatten':
            pass
        table = table_str


        cnt = 0
        while num_tokens_from_string(table, tokenizer) > args.max_length:
            table = " ".join(table.split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
            cnt += 200
        
        table_len = num_tokens_from_string(table, tokenizer)


        ###############################################    
        
        if args.task == 'qa':
            input = data['question']
            context = "Table is as follows. \n{} Question: {}".format(table, input)
        elif args.task == 'sql':
            input = data['sql']
            context = "Table is as follows. \n{} SQL Query: {} \nThe SQL execution result is".format(table, input)
        
        message = header + sys_prompt + context
        import pdb; pdb.set_trace()
        response, history = model.chat(tokenizer, message, history=[], do_sample=False)
        
        print(i, '[output]:', response, '[ground truth]:', data['answer'])
        
        # 查找"The answer is"在字符串中的位置
        start_index = response.find('The execution result is') + len('The execution result is')

        # 提取剩余的内容
        result = response[start_index:].strip('.')
        
        preds.append(result)
        golds.append(data['answer'])

        
        ###########################################
    em_score, f1_score = compute_emf1(preds, golds)

    print(f"em: {em_score}, f1: {f1_score}")
    
  
        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', choices=["qa", "sql"],  required=True)
    parser.add_argument('--format', choices=["markdown", "flatten"], required=True)
    parser.add_argument('--file_name', type=str, default='../TableQAEval.json')
    parser.add_argument('--max_length', type=int, default=6000)
    parser.add_argument('--max_new_tokens', type=int, default=1500)
    parser.add_argument('--mode', choices=["toy", "baby", "full"])
    parser.add_argument('--model_path', type=str, default='/home/lfy/PTM/chatglm2-6b')
    args = parser.parse_args()
    main(args)
    # CUDA_VISIBLE_DEVICES=0,3 python chatglm2_squall.py --task qa --format markdown --mode baby