import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
# Code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py

from functools import partial

import torch
import transformers
import transformers.models.llama.modeling_llama
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import argparse
from LEval_config import *
from tqdm import tqdm

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


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        header = (
            "A chat between a curious user and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        sys_prompt = get_sys_prompt(args, file_name)
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            # truncate documents
            while num_tokens_from_string(document, tokenizer) > args.max_length:
                document = " ".join(document.split()[:args.max_length - cnt])  # chunk the input len into 16k tokens
                cnt += 250

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name:
                    context = document + "\n\n" + inst
                    message = sys_prompt + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = header + " USER: " + + sys_prompt + context
                    message += " \nASSISTANT: "
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {} \nQuestion: {} "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nAnswer:"
                elif "coursera" in file_name:
                    context = "Document is as follows. {} Question: {} "
                    message = header + " USER: " + sys_prompt + context + " Please directly give answer without any additonal output or explanation "
                    message += " \nASSISTANT: "
                else:
                    context = "Document is as follows. {} \nInstruction: {} " + f"The suggested output length is around {len(out.split())} words. "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: "

                save_d['prompt'] = message.replace(document, "<long input>")
                inputs = tokenizer(message.format(document, inst), return_tensors="pt").to(device)
                prompt_length = inputs.input_ids.size()[-1]
                sample = model.generate(inputs.input_ids.to(model.device), do_sample=False, max_new_tokens=512, use_cache=True)[0]
                output = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)
                save_d[f'{open_source_model}_pred'] = output
                save_d['evaluation'] = d['evaluation']
                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
        fw.close()
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval"], required=True, help='metric name from choices')
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=7500, choices=[1500, 3500, 15000])
    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to use multiple choice')
    # for llama based model
    parser.add_argument('--scale', default='7b', choices=['7b', '13b'])
    parser.add_argument('--flash', action='store_true', help='set this if you want to use flash attention')
    args = parser.parse_args()
    key_data_pairs = {}

    # 7b / 13b
    model_path = f"lmsys/longchat-{args.scale}-16k"


    replace_llama_with_condense(8)
    open_source_model = f"longchat-{args.scale}-8k"
    if args.flash:
        replace_llama_attn_with_flash_attn()
        open_source_model = open_source_model + "-flash"
    if args.max_length == 1500:
        open_source_model = open_source_model.replace("8k", "2k")
    elif args.max_length == 3500:
        open_source_model = open_source_model.replace("8k", "4k")
    elif args.max_length == 15000:
        open_source_model = open_source_model.replace("8k", "16k")

    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    input(f"Your prediction file will be saved to: {data_save_path} \npress enter to confirm")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
