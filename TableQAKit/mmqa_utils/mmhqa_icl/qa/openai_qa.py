from typing import Dict, List, Union, Tuple
import time
# import api.api as api
import openai
from .prompt import PromptBuilder
import random
import os
import json

def parse_api_results(engine,result, result_idx_to_eid = None, verbose = False): # 转换 api 生成的结果格式
    if engine.lower() in ["llava","llava-13b","llava-7b"]:
        raise NotImplementedError
    else:
        # parse api results
        response_dict = dict()
        for idx, g in enumerate(result['choices']): # 枚举每个生成的结果
            try:
                text = g['text'] # 生成的文本
                logprob = sum(g['logprobs']['token_logprobs']) # 生成的文本的对数形式的概率, 为每个token的对数概率之和
                eid = result_idx_to_eid[idx] # 生成的文本对应的eid
                eid_pairs = response_dict.get(eid, None) # 生成的文本对应的eid的所有 text-probablity 对 (引用形式)
                if eid_pairs is None:
                    eid_pairs = []
                    response_dict[eid] = eid_pairs
                eid_pairs.append((text, logprob))

                if verbose:
                    print(text)

            except ValueError as e:
                if verbose:
                    print('---------- Error Msg --------')
                    print(e)
                    print(text)
                    print('-----------------------------')
                pass

        return response_dict


class Generator(object):
    """
    LLM generation wrapper.
    """

    def __init__(self, args, keys=None):
        self.args = args
        self.keys = keys
        self.current_key_id = 0
        self.prompt_builder = PromptBuilder(args)

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            n_shots: int,
            qtype: str
    ):
        """
        Build few-shot prompt for generation from file.
        """

        with open(os.path.join(self.args.template_path, file_path), 'r') as f:
            lines = f.readlines()
            
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        pre_prompt =  None # 乱序的时候忘了把这个加上, 结果效果差不多 ?
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            elif line == '\n' and pre_prompt == None: # 加上最开头的提示语
                pre_prompt = one_shot_prompt
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:n_shots]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt = '\n'.join(few_shot_prompt_list)
        return pre_prompt + '\n' + few_shot_prompt

    def build_generate_prompt(
            self,
            data_item: Dict,
            qtype: str,
            cot: bool = False
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            qtype=qtype, 
            cot=cot
            # only_title= True  # TEST: 生成的时候只用 title
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple], # [(g_eid, prompt)]
            args: Dict,
            verbose: bool = False
    ):
        """
        Generate one pass with llm according to the generation phase.
        """
        result_idx_to_eid = []
        for p in prompts:
            result_idx_to_eid.extend([p[0]] * args['n-samples']) # 生成 sampling_n 个答案？
        prompts = [p[1] for p in prompts]

        if verbose:
            print('\n', '*' * 20, 'LLM API Call', '*' * 20)
            for prompt in prompts:
                print(prompt)
                print('\n')
            print('- - - - - - - - - - ->>')

        status, result = self._call_llm_api(
            engine = self.args.engine,
            prompt=prompts,
            max_tokens = args['max_generation_tokens'],
            temperature = self.args.temperature,
            top_p = self.args.top_p,
            n = args['n-samples'],
            stop = self.args.stop_tokens
        )

        if not status:
            return False, None
        
        result_dict = parse_api_results(self.args.engine, result , result_idx_to_eid, verbose)
        """
        result_dict = {
            'eid': [(text1, logprob1),(text2,logprob2), ...... ],
            ......
        }
        """

        return status, result_dict
        

    def _call_llm_api(
            self,
            engine: str,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str]
    ):
        start_time = time.time()
        result = None
        while result is None:
            try:
                if engine.lower() in ["llava","llava-13b","llava-7b"]:
                    raise ValueError(f'{engine} is not supported now.')
                else:
                    key = self.keys[self.current_key_id]
                    self.current_key_id = (self.current_key_id + 1) % len(self.keys)
                    print(f"Using openai api key: {key}")
                    result = openai.Completion.create(
                        engine=engine,
                        prompt=prompt,
                        api_key=key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n, # number of samplesl,default to 20 in annotate
                        stop=stop,
                        logprobs=1
                    )
                    print('Openai api inference time:', time.time() - start_time)
                    return True, result
            except openai.error.InvalidRequestError as e:
                print(e, 'Modify the request.')
                return False, None
            except Exception as e:
                print(e, 'Retry.')
                time.sleep(5)
