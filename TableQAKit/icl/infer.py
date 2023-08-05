import json
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Union, Dict
from transformers import HfArgumentParser
from icl import GPT, fix_seed, Logger, print_time, ICLArguments
from icl import GPTDataSet


class ICL(ABC):
    def __init__(self, model: GPT, dataset: GPTDataSet):
        self.args = HfArgumentParser(ICLArguments).parse_args_into_dataclasses()[0]
        self.model = model
        self.dataset = dataset
        fix_seed(self.args.random_seed)

    def infer(self, demo_prefix: str, cot_trigger: str, answer_trigger: str):
        print('-' * 20)
        print(self.args)
        print('-' * 20)
        if not os.path.exists(self.args.logging_dir):
            os.mkdir(self.args.logging_dir)
        sys.stdout = Logger(os.path.join(self.args.logging_dir, print_time(1) + '.log'), sys.stdout)
        print_time()
        demo_input = self.create_demo_input(self.dataset.demos, demo_prefix, cot_trigger, answer_trigger)

        self.model.set_prompt(demo_input)
        for i, one in enumerate(self.dataset.data):
            if i < self.args.resume_id - 1:
                continue
            print('-' * 20)
            print("{}st data".format(i + 1))
            print("id: {}".format(one["id"]))
            data_input = self.create_data_input(one, cot_trigger, answer_trigger, self.args.truncation)
            print("demo input: \n{}".format(demo_input))
            print("data input: \n{}".format(data_input))
            gpt_out = self.model.getResponse(
                data_input,
                self.args.temperature,
                self.args.max_length,
                self.args.api_time_interval
            )
            print("gpt output: {}".format(gpt_out))
            if answer_trigger not in gpt_out:
                answer_extract = self.model.getResponse(
                    data_input + gpt_out + answer_trigger,
                    self.args.temperature,
                    self.args.max_length,
                    self.args.api_time_interval
                )
            else:
                answer_extract = gpt_out.split(answer_trigger)[-1]
            answer = self.answer_post_proc(answer_extract)
            print("pred answer: {}".format(answer))
            self.save_prediction(self.args.output_path, one, answer)

    @staticmethod
    def remove_space(text_in):
        res = []
        for tmp in text_in.split(" "):
            if tmp != "":
                res.append(tmp)
        return " ".join(res)

    def table_flatten(self, heads: List[str], row: List[str]):
        res = ""
        if heads[0]:
            res += (heads[0] + " ")
        for head, cell in zip(heads[1:], row[1:]):
            res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
        res = self.remove_space(res)
        return res.strip()

    @staticmethod
    def table_markdown(row: List[str]):
        res = ""
        for cell in row:
            if cell != "":
                res += ("| " + cell + " ")
            else:
                res += "| - "
        return res + "|"

    @abstractmethod
    def create_demo_input(
            self,
            demos: List[Dict[str, str]],
            demo_prefix: str,
            cot_trigger: str,
            answer_trigger: str,
    ) -> Union[str, List[Dict[str, str]]]:
        """
        According to the demos from the dataset, create the demo that will be input to the openai model
        :param demo_prefix: The prefix of demo_input like "Reading the texts and tables and try your best to answer the question."
        :param demos: demos list from the dataset.create_demos()
        :param cot_trigger: the prompt to trigger cot like "Let's think step by step."
        :param answer_trigger: the prompt to trigger the answer like "Therefore, the answer is "
        :return: str for davinci like
        "Q: demo1
         A: answer1

         Q: demo2
         A: ……"

         and List[Dict] for turbo like
         [{"role": "user", "content": demo1}, {"role": "assistant", "content": answer1},
         {"role": "user", "content": demo2}, {"role": "assistant", "content": answer2},
         ……]
        """
        pass

    @abstractmethod
    def create_data_input(
            self,
            data: Dict[str, str],
            cot_trigger: str,
            answer_trigger: str,
            truncation: int
    ) -> str:
        """
        According to the data from the dataset, create the data that will be input to the openai model
        :param data: one data from the dataset
        :param cot_trigger: the prompt to trigger cot like "Let's think step by step."
        :param answer_trigger: the prompt to trigger the answer like "Therefore, the answer is "
        :param truncation: cut off the max length of content
        :return: str for all Openai model like
        "Q: paragraphs:……
        tables:……
        question:……
        A: cot_trigger
        """
        pass

    @abstractmethod
    def answer_post_proc(self, answer_extract: str) -> str:
        """
        Post-processing the answer from gpt output
        :param answer_extract: answer from gpt_output.split(answer_trigger)[-1]
        :return: processing the answer if needed.
        """
        pass

    @abstractmethod
    def save_prediction(
            self,
            output_path: str,
            data: Dict[str, str],
            answer: str
    ) -> None:
        """
        Saving the answer with the form that you are fond of
        :param output_path: your arg input
        :param data: one data from the dataset
        :param answer: the answer after post-processing.
        :return: None
        """
        pass


class turboICL(ICL):
    def create_demo_input(self, demos: List[dict], demo_prefix: str, cot_trigger: str, answer_trigger: str) -> Union[str, List[Dict[str, str]]]:
        demo_input = [
            {"role": "user", "content": demo_prefix}]
        for demo in demos:
            demo_input.append({
                "role": "user",
                "content": demo["question"]
            })
            demo_input.append({
                "role": "assistant",
                "content": cot_trigger + demo["rationale"] + answer_trigger + demo["answer"]
            })
        return demo_input

    def create_data_input(self, data: dict, cot_trigger: str, answer_trigger: str, truncation: int) -> str:
        heads = []
        content = ""
        if data["texts"] is not None and len(data["texts"]):
            content += "paragraphs:\n"
            for text in data["texts"]:
                content += (text + "\n")
        if data["rows"] is not None and len(data["rows"]):
            content += "table:\n"
            for i, row in enumerate(data["rows"]):
                if self.args.use_table_markdown:
                    content += (self.table_markdown(row) + "\n")
                elif self.args.use_table_flatten:
                    if i == 0:
                        heads = row
                    else:
                        content += (self.table_flatten(heads, row) + "\n")
                else:
                    raise NotImplementedError
        question = "question:\n" + data["question"] + "\n" + cot_trigger
        if len(content) > truncation - len(question):
            content = content[:truncation - len(question) - 1] + '\n'
        content += question
        return content

    def answer_post_proc(self, answer_extract: str) -> str:
        return answer_extract

    def save_prediction(self, output_path: str, data: dict, answer: str) -> None:
        if not os.path.isfile(output_path):
            with open(output_path, 'w') as file:
                json.dump([], file)
        output_data = {
            "uid": data["id"],
            "predicted_program": [],
            "predicted_ans": answer
        }
        with open(output_path, 'r+') as file:
            file_data = json.load(file)
            file_data.append(output_data)
            file.seek(0)
            json.dump(file_data, file, indent=4)


class davinciICL(ICL):
    def create_demo_input(self, demos: List[dict], demo_prefix: str, cot_trigger: str, answer_trigger: str) -> Union[str, List[Dict[str, str]]]:
        demo_input = demo_prefix + "\n"
        for demo in demos:
            demo_input += ("Q: " + demo["question"] + "\n" +
                           "A: " + cot_trigger + demo["rationale"] +
                           answer_trigger + demo["answer"] + "\n\n")
        return demo_input

    def create_data_input(self, data: dict, cot_trigger: str, answer_trigger: str, truncation: int) -> str:
        heads = []
        content = "Q: \n"
        if data["texts"] is not None and len(data["texts"]):
            content += "paragraphs:\n"
            for text in data["texts"]:
                content += (text + "\n")
        if data["rows"] is not None and len(data["rows"]):
            content += "table:\n"
            for i, row in enumerate(data["rows"]):
                if self.args.use_table_markdown:
                    content += (self.table_markdown(row) + "\n")
                elif self.args.use_table_flatten:
                    if i == 0:
                        heads = row
                    else:
                        content += (self.table_flatten(heads, row) + "\n")
                else:
                    raise NotImplementedError
        question = "question:\n" + data["question"] + "\n" + "A: " + cot_trigger
        if len(content) > truncation - len(question):
            content = content[:truncation - len(question) - 1] + '\n'
        content += question
        return content

    def answer_post_proc(self, answer_extract: str) -> str:
        return answer_extract

    def save_prediction(self, output_path: str, data: dict, answer: str) -> None:
        if not os.path.isfile(output_path):
            with open(output_path, 'w') as file:
                json.dump([], file)
        output_data = {
            "uid": data["id"],
            "predicted_program": [],
            "predicted_ans": answer
        }
        with open(output_path, 'r+') as file:
            file_data = json.load(file)
            file_data.append(output_data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
