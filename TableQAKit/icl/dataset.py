import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict


class GPTDataSet(ABC):
    def __init__(self, data_path: str, demo_path: Optional[str]):
        self.data_path = data_path
        self.demo_path = demo_path
        self.data = self.read_data(self.data_path)
        self.demos = self.read_demos(self.demo_path)

    @abstractmethod
    def read_data(self, data_path: str) -> List[Dict[str, any]]:
        """
        Read the data from the specified data path and preprocess it into the following format:
        :param data_path: The path of the data file.
        :return: A list of dictionaries, each containing the following keys:
            - "id": The unique identifier of the data.
            - "question": The question associated with the data.
            - "texts": A list of strings representing the texts associated with the data.
            - "rows": A list representing the tables' rows associated with the data.
        """
        pass

    @abstractmethod
    def read_demos(self, demo_path: str) -> List[Dict[str, any]]:
        """
        Read the demonstration data from the specified demo path and preprocess it into the following format:
        :param demo_path: The path of the demonstration file.
        :return: A list of dictionaries, each containing the following keys:
            - "question": The question of the demonstration.
            - "rationale": The CoT (Causes of Truth) rationale of the demonstration.
            - "answer": The golden answer of the demonstration.
        """
        pass


class MultiHiertt(GPTDataSet):
    def read_data(self, data_path: str) -> List[Dict]:
        data_list = []
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        for one in data:
            texts = []
            rows = []
            for text_evidence in one["qa"]["text_evidence"]:
                texts.append(one["paragraphs"][text_evidence])

            for table_evidence in one["qa"]["table_evidence"]:
                rows.append(one["table_description"][table_evidence])
            data_list.append({
                "id": one["uid"],
                "question": one["qa"]["question"],
                "texts": texts,
                "rows": rows,
            })
        return data_list

    def read_demos(self, demo_path: str) -> List[Dict]:
        if demo_path is not None:
            return json.load(
                open(demo_path, 'r', encoding='utf-8')
            )
        else:
            return []


class UnifiedSKG(GPTDataSet):
    def read_data(self, data_path: str) -> List[Dict[str, any]]:
        data_list = []
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        for one in data:
            data_list.append({
                "id": self.get_id(one),
                "question": one["question"],
                "texts": None,
                "rows": self.get_rows(one),
            })
        return data_list

    @abstractmethod
    def get_id(self, one) -> Optional[str]:
        pass

    @abstractmethod
    def get_rows(self, one) -> List[str]:
        pass

    def read_demos(self, demo_path: str) -> List[Dict[str, any]]:
        if demo_path is not None:
            return json.load(
                open(demo_path, 'r', encoding='utf-8')
            )
        else:
            return []


class Wikisql(UnifiedSKG):
    def get_id(self, one) -> Optional[str]:
        return None

    def get_rows(self, one) -> List[str]:
        return [one["table"]["header"]] + one["table"]["rows"]


class Wikitq(UnifiedSKG):
    def get_id(self, one) -> Optional[str]:
        return one["id"]

    def get_rows(self, one) -> List[str]:
        return [one["table"]["header"]] + one["table"]["rows"]


class FinQA(GPTDataSet):
    def read_data(self, data_path: str) -> List[Dict[str, any]]:
        data_list = []
        data = json.load(
            open(data_path, 'r', encoding='utf-8')
        )
        for one in data:
            texts = []
            for k, v in one["qa"]["gold_inds"].items():
                if "text" in k:
                    texts.append(v)
            data_list.append({
                "id": one["id"],
                "question": one["qa"]["question"],
                "texts": texts,
                "rows": one["table"],
            })
        return data_list

    def read_demos(self, demo_path: str) -> List[Dict[str, any]]:
        if demo_path is not None:
            return json.load(
                open(demo_path, 'r', encoding='utf-8')
            )
        else:
            return []
