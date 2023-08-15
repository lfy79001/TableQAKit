import json, os
import openai
import argparse
import pandas as pd
import tiktoken
import time
import sys
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM,AutoModelForCausalLM
from transformers.generation import GenerationConfig
import torch
sys.path.append('./')
sys.path.append('../')
sys.path.append('../Evaluation')

# from Evaluation.em import compute_exact_match
# from Evaluation.f1 import compute_f1
from Evaluation.emf1 import compute_emf1


def num_tokens_from_string(table, tokenizer):
    return len(tokenizer.tokenize(table))

def generate_prompt_format(args):
    if args.task == 'qa':
       return 'Answer the question based on the given table data. Only give me the answer and do not output any other words. Table:\n\n{}\n\nAnswer the question based on the given table data. Only give me the answer and do not output any other words. The following are some examples.\n\n{}'
    elif args.task == 'sql':
        return "You are an SQL executor, you need to execute SQL based on the give table and SQL statement to obtain the execution results. Only give me the execution results and do not output any other words. \nTable: {}\nNow you need to execute SQL based on the given table and SQL statement to obtain the execution result. Only give me the result and do not output any other words or SQL statement.\nThe following are some examples.\n\n{}"

def generate_table(data):
    header = data['header']
    contents = data['contents']
    header_string = "col : " + " | ".join(data["header"]) + " "
    value_string = ""
    for i, row in enumerate(data['contents']):
        value_string += "row " + str(i+1) + " : "
        row_cell_values = [str(cell_value) if isinstance(cell_value, int) else cell_value.lower()
                           for cell_value in row]
        value_string += " | ".join(row_cell_values) + " "
    output_string = header_string + value_string
    return output_string

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True)
    model = model.eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)

    input_format = generate_prompt_format(args)
    
    with open(args.file_name, 'r') as file:
        json_data = json.load(file)    
    
    if args.mode == 'toy':
        json_data = json_data[:100]

    outputs = []
    for i, data in enumerate(json_data):
        if args.format == 'markdown':
            df = pd.DataFrame(data['contents'], columns=data['header'])
            table_str = df.to_markdown(headers=data['header'], tablefmt="pipe")
            # table_str = generate_table(data)
        elif args.format == 'flatten':
            pass
        table = table_str

        cnt = 0
        while num_tokens_from_string(table, tokenizer) > args.max_length:
            table = " ".join(table.split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
            cnt += 500
        
        table_len = num_tokens_from_string(table, tokenizer)
        

        ################    #################    
        messages = []
        
        if args.task == 'qa':
            examples = data['examples']
            fewshot_input = ""
            for example in examples[:args.n_shot]:
                fewshot_input += "Question:{}\nAnswer:{}\n".format(example['question'], example['answer'])
            fewshot_input += f"Question:{examples[args.n_shot]['question']}\nAnswer:"
            gold_answer = examples[args.n_shot]['answer']
            input_prompt = input_format.format(table, fewshot_input)
        elif args.task == 'sql':
            examples = data['examples']
            fewshot_input = ""
            for example in examples[:args.n_shot]:
                fewshot_input += "SQL:{}\nAnswer:{}\n".format(example['sql'], example['answer'])
            fewshot_input += f"SQL:{examples[args.n_shot]['sql']}\nAnswer:"
            gold_answer = examples[args.n_shot]['answer']
            input_prompt = input_format.format(table, fewshot_input)
            

        result, history = model.chat(tokenizer, input_prompt, history=[], do_sample=False)

        print(i, '[output]:', result, '[ground truth]:', gold_answer)

        outputs.append({'table_id':data['table_id'], 'pred':result, 'gold': gold_answer})  
        
    save_path = f'../results/fewshot/qwen_chat_{args.task}_{args.n_shot}_{args.mode}_{len(json_data)}.json'
    if 'wiki' in args.file_name:
        save_path = f'../results/fewshot/qwen_chat_{args.task}_{args.n_shot}_{args.mode}_{len(json_data)}_wiki.json'
    with open(save_path, 'w') as f:
        json.dump(outputs, f, indent=2)
    print(f'saving to {save_path}')



    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', choices=["qa", "sql"],  required=True)
    parser.add_argument('--format', choices=["markdown", "flatten"], required=True)
    parser.add_argument('--file_name', type=str, default='../data/sql_fourshot.json')
    parser.add_argument('--max_length', type=int, default=3500)
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--mode', choices=["toy", "baby", "full"])
    parser.add_argument('--model_path', type=str, default='/home/lfy/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/8130170c072b681f3c0e5664422c8b8d2779b2c6')
    parser.add_argument('--n_shot', type=int, default=4)
    args = parser.parse_args()
    main(args)
    # python qwen_chat_fewshot.py --task sql --format markdown --n_shot 4 --mode full