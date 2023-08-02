from typing import Union, Dict
import openai
import time
from abc import ABC, abstractmethod


class GPT(ABC):
    def __init__(self, key: str):
        self.key = key
        openai.api_key = self.key
        self.prompt = None

    def set_prompt(self, prompt: Union[str, Dict[str, str]]):
        self.prompt = prompt

    @abstractmethod
    def getResponse(self, content: str, temperature: float, max_length: int, api_time_interval: float) -> str:
        """
        Get response from the OpenAI model.

        :param content: The input text.
        :param temperature: The temperature for the OpenAI model.
        :param max_length: The maximum length for the OpenAI model response.
        :param api_time_interval: The sleep time for the API (1.0 for text_davinci_003).
        :return: The response text from the OpenAI model.
        """
        pass


class turbo(GPT):
    def addText(self, content: str):
        self.prompt.append({"role": "user", "content": content})

    def getResponse(self, content: str, temperature: float, max_length: int, api_time_interval: float) -> str:
        time.sleep(api_time_interval)
        self.addText(content)
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=self.prompt,
            temperature=temperature
        )
        resText = response.choices[0].message.content
        self.deleteText()
        return resText

    def deleteText(self):
        self.prompt.pop()


class text_davinci_003(GPT):
    def getResponse(self, content: str, temperature: float, max_length: int, api_time_interval: float) -> str:
        time.sleep(api_time_interval)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.prompt + content,
            max_tokens=max_length,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response["choices"][0]["text"]
