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
import transformers
from Evaluation.emf1 import compute_emf1, compute_exact, compute_f1, save_results
from transformers import pipeline



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
    config = transformers.AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda' # For fast initialization directly on GPU!

    model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model_path,
    config=config,
    torch_dtype=torch.bfloat16, # Load model weights in bfloat16
    trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    model = model.eval()


    
    with open(args.file_name, 'r') as file:
        json_data = json.load(file)    
    
    if args.mode == 'toy':
        json_data = json_data[:10]
    elif args.mode == 'baby':
        json_data = json_data[100:200]

    preds = []
    golds = []
    sources = []
    for i, data in enumerate(json_data):
        
        if args.format == 'markdown':
            df = pd.DataFrame(data['contents'], columns=data['header'])
            table_str = df.to_markdown(headers=data['header'], tablefmt="pipe")
        elif args.format == 'flatten':
            pass
        table = table_str

        sys_prompt = generate_sys_prompt(data['source'])


        cnt = 0
        if data['source'] == 'structured':
            while num_tokens_from_string(table, tokenizer) > args.max_length:
                table = " ".join(table.split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
                cnt += 200
        else:
            count = 0
            while num_tokens_from_string(table + '\n ' + data['passage'], tokenizer) > args.max_length:
                data['passage'] = " ".join(data['passage'].split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
                cnt += 200
                count += 1
                if count > 50:
                    break
            if count > 50:
                cnt = 0
                while num_tokens_from_string(table, tokenizer) > args.max_length:
                    table = " ".join(table.split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
                    cnt += 200

        ###############################################    
        
        question = data['question']
        
        if data['source'] == 'structured':
            context = "Table is as follows. \n{} Question: {}".format(table, question)
        else:
            context = "Table is as follows. \n{} \n Passage is as follows \n {}Question: {}".format(table, data['passage'], question)
        
        message = sys_prompt + context + "The answer to this question is: "

        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

        with torch.autocast('cuda', dtype=torch.bfloat16):
            response = pipe(message,
                    max_new_tokens=100,
                    do_sample=True,
                    use_cache=True)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            response = pipe(message, max_new_tokens=512, do_sample=False, use_cache=True)[0]['generated_text'][len(message):]

        print(i, '[output]:', response, '[ground truth]:', data['answer'])
        
        # 查找"The answer is"在字符串中的位置
        start_index = response.find('The answer is') + len('The answer is')
        # 提取剩余的内容
        if start_index != -1:
            result = response[start_index:].strip('.')
        else:
            result = response
        
        preds.append(result)
        golds.append(data['answer'])
        sources.append(data['source'])
        
        ###########################################
    em_score, f1_score = compute_emf1(preds, golds)
    print(f"total: em {em_score}, f1: {f1_score} ")
    numerical1, multihop1, structured1, total1 = [], [], [], []
    numerical2, multihop2, structured2, total2 = [], [], [], []
    for pred, gold, source in zip(preds, golds, sources):
        em = compute_exact(str(pred), str(gold))
        f1 = compute_f1(str(pred), str(gold))
        if source == 'numerical':
            numerical1.append(em)
            numerical2.append(f1)
        elif source == 'multihop':
            multihop1.append(em)
            multihop2.append(f1)
        elif source == 'structured':
            structured1.append(em)
            structured2.append(f1)
        total1.append(em)
        total2.append(f1)
    
    if len(numerical1) > 0: print(f"numerical: em {sum(numerical1) / len(numerical1) * 100}, f1: {sum(numerical2) / len(numerical2) * 100} {len(numerical1)}")
    if len(multihop1) > 0: print(f"multihop: em {sum(multihop1) / len(multihop1) * 100}, f1: {sum(multihop2) / len(multihop2) * 100} {len(multihop1)}")
    if len(structured1) > 0: print(f"structured: em {sum(structured1) / len(structured1) * 100}, f1: {sum(structured2) / len(structured2) * 100} {len(structured1)}")
    print(f"total: em {sum(total1) / len(total1) * 100}, f1: {sum(total2) / len(total2) * 100} {len(total2)}")
    save_results(preds, '../Results/mpt.txt')
    
  
        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--format', choices=["markdown", "flatten"], required=True)
    parser.add_argument('--file_name', type=str, default='../TableQAEval.json')
    parser.add_argument('--max_length', type=int, default=7000)
    parser.add_argument('--max_new_tokens', type=int, default=1500)
    parser.add_argument('--mode', choices=["toy", "baby", "full"])
    parser.add_argument('--model_path', type=str, default='mosaicml/mpt-7b-storywriter')
    args = parser.parse_args()
    main(args)
    # CUDA_VISIBLE_DEVICES=0,1 python mpt-table.py --format markdown --mode baby