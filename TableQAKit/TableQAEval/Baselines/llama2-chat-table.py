import json, os
import openai
import argparse
import pandas as pd
import tiktoken
import time
import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
sys.path.append('./')
sys.path.append('../')
sys.path.append('../Evaluation')

from Evaluation.emf1 import compute_emf1


def generate_sys_prompt(source):
    if source == 'multihop':
        return 'Give you a table, and the paragraphs related to it, and the corresponding questions. You need to answer the questions according to the table and paragraph. Make sure the response is end with The answer is _ .'
    elif source == 'numerical':
        return 'Give you a table, and the paragraphs related to it, and the corresponding questions. You need to reason and calculate numerical problems to obtain answers. Make sure the response is end with The answer is _ .'
    elif source == 'structured':
        return 'Give you a table and the corresponding question. You need to answer the questions according to the table. Make sure the response is end with The answer is _ '


def num_tokens_from_string(table, tokenizer):
    return len(tokenizer.tokenize(table))


def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    
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
        
        sys_prompt = generate_sys_prompt(data['source'])


        cnt = 0
        while num_tokens_from_string(table, tokenizer) > args.max_length:
            table = " ".join(table.split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
            cnt += 200
        
        table_len = num_tokens_from_string(table, tokenizer)

        ###############################################    
        
        question = data['question']
        
        if data['source'] == 'structured':
            context = "Table is as follows. \n{} Question: {}".format(table, question)
        else:
            context = "Table is as follows. \n{} \n Passage is as follows \n {}Question: {}".format(table, data['passage'], question)

        message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
        
        inputs = tokenizer(message, return_tensors="pt").to(0)

        sample = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        prompt_length = inputs.input_ids.size()[-1]

        output = tokenizer.decode(sample[0][prompt_length:])
        response = output.replace('</s>', '')
        
        print(i, '[output]:', response, '[ground truth]:', data['answer'])
        
        if data['source'] != 'numerical':
            # 查找"The answer is"在字符串中的位置
            start_index = response.find('The answer is') + len('The answer is')
            # 提取剩余的内容
            result = response[start_index:].strip('.')
        else:
            result = response
        
        preds.append(result)
        golds.append(data['answer'])

        ###########################################
    em_score, f1_score = compute_emf1(preds, golds)

    print(f"em: {em_score}, f1: {f1_score}")
    
  
        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--format', choices=["markdown", "flatten"], required=True)
    parser.add_argument('--file_name', type=str, default='../TableQAEval.json')
    parser.add_argument('--max_length', type=int, default=3500)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--mode', choices=["toy", "baby", "full"])
    parser.add_argument('--model_path', type=str, default='/home/lfy/PTM/Llama-2-7b-chat-hf')
    args = parser.parse_args()
    main(args)
    # CUDA_VISIBLE_DEVICES=4,5,6,7,8 python llama2-chat-table.py --format markdown --mode baby