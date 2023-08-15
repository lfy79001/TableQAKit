import json, os
import openai
import argparse
import pandas as pd
import tiktoken
import time
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../Evaluation')

# from Evaluation.em import compute_exact_match
# from Evaluation.f1 import compute_f1
from Evaluation.emf1 import compute_emf1


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_prompt_format(args):
    if args.task == 'qa':
       return 'Answer the question based on the given table data. Only give me the answer and do not output any other words. Table:\n\n{}\n\nAnswer the question based on the given table data. Only give me the answer and do not output any other words.The following are some examples.\n\n{}'
    elif args.task == 'sql':
        return "You are an SQL executor, you need to execute SQL based on the give table and SQL statement to obtain the execution results. Only give me the execution results and do not output any other words. \nTable: {}\nNow you need to execute SQL based on the given table and SQL statement to obtain the execution result. Only give me the result and do not output any other words or SQL statement.\nThe following are some examples.\n\n{}"

def main(args):
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
        elif args.format == 'flatten':
            pass
        table = table_str

        cnt = 0
        while num_tokens_from_string(table, "gpt-3.5-turbo") > args.max_length:
            table = " ".join(table.split()[:args.max_length - cnt]) # chunk the input len into 16k tokens
            cnt += 500
        

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

        messages.append({"role": "user", "content": input_prompt})

        for _ in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k-0613",
                    messages=messages, 
                    max_tokens=args.max_new_tokens,
                    temperature=0.0001,
                )  # get response
                result = response['choices'][0]['message']['content']
                result = result.strip()  # get the paraphrased answer

                print(i, '[output]:', result, '[ground truth]:', gold_answer)
                
                break
            except Exception as e:  # add some logit here for retry
                if isinstance(e, KeyboardInterrupt):
                    raise e
                time.sleep(0.1)
        outputs.append({'table_id':data['table_id'], 'pred':result, 'gold': gold_answer})  
        time.sleep(1.0)
        ###########################################
    save_path = f'../results/fewshot/turbo16k_{args.task}_{args.n_shot}_{args.mode}.json'
    if 'wiki' in args.file_name:
        save_path = f'../results/fewshot/turbo16k_{args.task}_{args.n_shot}_{args.mode}_wiki.json'
    
    with open(save_path, 'w') as f:
        json.dump(outputs, f, indent=2)
    print(f"saving to {save_path}")

    

if __name__ == '__main__':
    openai.api_key = ""
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', choices=["qa", "sql"],  required=True)
    parser.add_argument('--format', choices=["markdown", "flatten"], required=True)
    parser.add_argument('--file_name', type=str, default='../data/wikisql_fourshot.json')
    parser.add_argument('--max_length', type=int, default=15000)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--mode', choices=["toy", "baby", "full"])
    parser.add_argument('--n_shot', type=int, default=4)
    args = parser.parse_args()
    main(args)
    # python turbo16k.py --task sql --format markdown --mode toy
    # python turbo16k.py --task qa --format markdown --n_shot 4 --mode toy