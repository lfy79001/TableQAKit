import json, os
import openai
import argparse
import pandas as pd
import tiktoken
import time
import sys
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
from functools import partial
sys.path.append('./')
sys.path.append('../')
sys.path.append('../Evaluation')

from Evaluation.emf1 import compute_emf1, compute_exact, compute_f1, save_results


def generate_sys_prompt(source):
    if source == 'multihop':
        return 'Give you a table, and the paragraphs related to it, and the corresponding questions. You need to answer the questions according to the table and paragraph. '
    elif source == 'numerical':
        return 'Give you a table, and the paragraphs related to it, and the corresponding questions. You need to reason and calculate numerical problems to obtain answers.'
    elif source == 'structured':
        return 'Give you a table and the corresponding question. You need to answer the questions according to the table. '


def num_tokens_from_string(table, tokenizer):
    return len(tokenizer.tokenize(table))



class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(
        self, dim, ratio, max_position_embeddings=2048, base=10000, device=None
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.ratio = ratio
        max_position_embeddings *= ratio
        self.max_seq_len_cached = max_position_embeddings
        # print(f"Monkey Patching condense ratio {ratio}")
        t = (
            torch.arange(
                self.max_seq_len_cached,
                device=self.inv_freq.device,
                dtype=self.inv_freq.dtype,
            )
            / ratio
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = (
                torch.arange(
                    self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
                )
                / self.ratio
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def replace_llama_with_condense(ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(
        CondenseRotaryEmbedding, ratio=ratio
    )



def main(args):
    
    replace_llama_with_condense(8)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()

    header = ""
    
    with open(args.file_name, 'r') as file:
        json_data = json.load(file)    
    
    if args.mode == 'toy':
        json_data = json_data[:10]
    elif args.mode == 'baby':
        json_data = json_data[100:1000]

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
            context = "Question: {} \nTable is as follows. \n{} \nQuestion: {}".format(question, table, question)
        else:
            context = "Question: {} \nTable is as follows. \n{} \n Paragraph is as follows \n {}\nQuestion: {}".format(question, table, data['passage'], question)

        message = header + " USER: " + sys_prompt + context + "Make sure the response is end with The answer is _ ." +   " \nASSISTANT: "

        inputs = tokenizer(message, return_tensors="pt").to(0)
        prompt_length = inputs.input_ids.size()[-1]
        sample = model.generate(inputs.input_ids.to(model.device), do_sample=False, max_new_tokens=20, use_cache=True)[0]
        response = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)
        
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
    save_results(preds, '../Results/longchat.txt')
    
  
        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--format', choices=["markdown", "flatten"], required=True)
    parser.add_argument('--file_name', type=str, default='../TableQAEval.json')
    parser.add_argument('--max_length', type=int, default=5000)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--mode', choices=["toy", "baby", "full"])
    parser.add_argument('--model_path', type=str, default='/home/lfy/PTM/longchat-7b-16k')
    args = parser.parse_args()
    main(args)
    # CUDA_VISIBLE_DEVICES=1,2,7,8 python longchat-table.py --format markdown --mode baby