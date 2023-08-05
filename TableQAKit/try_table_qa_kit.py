from typing import List, Optional, Union, Dict
from icl import turbo, text_davinci_003, FinQA, Wikisql, Wikitq, TAT_QA

DATA_PATH = {
    "finqa": {
        "train": "datasets/finqa/train.json",
        "dev": "datasets/finqa/dev.json",
        "eval": None,
        "test": None
    },
    "tatqa": {
        "train": "datasets/tatqa/train.json",
        "dev": "datasets/tatqa/dev.json",
        "eval": None,
        "test": "datasets/tatqa/test.json"
    },
    "wikisql": {
        "train": "datasets/wikisql/train.json",
        "dev": "datasets/wikisql/dev.json",
        "eval": None,
        "test": "datasets/wikisql/test.json"
    },
    "wikitq": {
        "train": "datasets/wikitq/train.json",
        "dev": "datasets/wikitq/dev.json",
        "eval": None,
        "test": "datasets/wikitq/test.json"
    },
    "spreadsheetqa": {
        "train": "datasets/spreadsheetqa/train.json",
        "dev": "datasets/spreadsheetqa/dev.json",
        "eval": None,
        "test": "datasets/spreadsheetqa/test.json"
    }
}

DATA_CLASS = {
    "finqa": FinQA,
    "tatqa": TAT_QA,
    "wikisql": Wikisql,
    "wikitq": Wikitq,
    "spreadsheetqa": FinQA
}


class TableQAKitDemo:
    def __init__(self, model_type: str, key: str):
        self.model_type = model_type
        if model_type == "text_davinci_003":
            self.model = text_davinci_003(key=key)
        elif model_type == "turbo":
            self.model = turbo(key=key)
        else:
            raise NotImplementedError

    @staticmethod
    def table_markdown(row: List[str]):
        res = ""
        for cell in row:
            if cell != "":
                res += ("| " + cell + " ")
            else:
                res += "| - "
        return res + "|"

    @staticmethod
    def get_demo_by_data(data: str) -> Dict:
        demo_dict = {
            "finqa": {
                "question": "table:\n| company | payments volume ( billions ) | total volume ( billions ) | total transactions ( billions ) | cards ( millions ) |\n| visa inc. ( 1 ) | $ 2457 | $ 3822 | 50.3 | 1592 |\n| mastercard | 1697 | 2276 | 27.0 | 916 |\n| american express | 637 | 647 | 5.0 | 86 |\n| discover | 102 | 119 | 1.6 | 57 |\n| jcb | 55 | 61 | 0.6 | 58 |\n| diners club | 29 | 30 | 0.2 | 7 |\nquestion:\nwhat is the average payment volume per transaction for american express?",
                "rationale": "To find the average payment volume per transaction for American Express, we need to divide the total payment volume by the total number of transactions. From the table, we can see that the payment volume for American Express is $637 billion and the total number of transactions is 5.0 billion. To find the average payment volume per transaction, we divide $637 billion by 5.0 billion: Average payment volume per transaction = $637 billion / 5.0 billion = $127.4.",
                "answer": "127.4"
            },
            "tatqa":
                {
                    "question": "Texts:\nThe following table provides the weighted average actuarial assumptions used to determine net periodic benefit costfor years ended:\nFor domestic plans, the discount rate was determined by comparison against the FTSE pension liability index for AA rated corporate instruments. The Company monitors other indices to assure that the pension obligations are fairly reported on a consistent basis. The international discount rates were determined by comparison against country specific AA corporate indices, adjusted for duration of the obligation.\nThe periodic benefit cost and the actuarial present value of projected benefit obligations are based on actuarial assumptions that are reviewed on an annual basis. The Company revises these assumptions based on an annual evaluation of longterm trends, as well as market conditions that may have an impact on the cost of providing retirement benefits.\ntable:\n| - | Domestic | - | International | - |\n| - | September 30, | - | September 30, | - |\n| - | 2019 | 2018 | 2019 | 2018 |\n| Discount rate | 4.00% | 3.75% | 1.90% | 2.80% |\n| Expected return on plan assets | - | - | 3.40% | 3.70% |\n| Rate of compensation increase | - | - | - - % | - - % |\nquestion:\nWhat is the year on year percentage change in domestic discount rate between 2018 and 2019?",
                    "rationale": "The domestic discount rate for 2018 was 3.75%, and the domestic discount rate for 2019 was 4.00%. To calculate the year on year percentage change, we need to subtract the 2018 rate from the 2019 rate and divide by the 2018 rate. ",
                    "answer": "6.67%"
                },
            "wikisql":
                {
                    "question": "table:\n| Overall Pick # | AFL Team | Player | Position | College |\n| 4 | Miami Dolphins | Bob Griese | Quarterback | Purdue |\n| 5 | Houston Oilers | George Webster 1 | Linebacker | Michigan State |\n| 6 | Denver Broncos | Floyd Little | Running Back | Syracuse |\n| 12 | New York Jets | Paul Seiler | Offensive Guard | Notre Dame |\n| 14 | San Diego Chargers | Ron Billingsley | Defensive Tackle | Wyoming |\n| 17 | Oakland Raiders | Gene Upshaw | Offensive Guard | Texas A&I |\n| 21 | Boston Patriots | John Charles | Defensive Back | Purdue |\n| 22 | Buffalo Bills | George Daney | Offensive Guard | Arizona State |\n| 23 | Houston Oilers | Tom Regner | Offensive Guard | Notre Dame |\nquestion:\nWhat is the position of john charles?",
                    "rationale": "The table provided contains information about the overall pick number, AFL team, player, position, and college of each player. To answer the question, we need to look for the row that contains information about John Charles. We can see that the row with Overall Pick # 21 contains information about John Charles. ",
                    "answer": "Defensive Back"
                },
            "wikitq":
                {
                    "question": "table:\n| # | Player | Goals | Caps | Career |\n| 1 | Landon Donovan | 57 | 155 | 2000–present |\n| 2 | Clint Dempsey | 36 | 103 | 2004–present |\n| 3 | Eric Wynalda | 34 | 106 | 1990–2000 |\n| 4 | Brian McBride | 30 | 95 | 1993–2006 |\n| 5 | Joe-Max Moore | 24 | 100 | 1992–2002 |\n| 6T | Jozy Altidore | 21 | 67 | 2007–present |\n| 6T | Bruce Murray | 21 | 86 | 1985–1993 |\n| 8 | Eddie Johnson | 19 | 62 | 2004–present |\n| 9T | Earnie Stewart | 17 | 101 | 1990–2004 |\n| 9T | DaMarcus Beasley | 17 | 114 | 2001–present |\nquestion:\nwho scored more goals: clint dempsey or eric wynalda?",
                    "rationale": "According to the table, Clint Dempsey scored 36 goals and Eric Wynalda scored 34 goals. Therefore, Clint Dempsey scored more goals than Eric Wynalda. ",
                    "answer": "Clint Dempsey"
                },
            "spreadsheetqa": {
                "question": "table:\n| company | payments volume ( billions ) | total volume ( billions ) | total transactions ( billions ) | cards ( millions ) |\n| visa inc. ( 1 ) | $ 2457 | $ 3822 | 50.3 | 1592 |\n| mastercard | 1697 | 2276 | 27.0 | 916 |\n| american express | 637 | 647 | 5.0 | 86 |\n| discover | 102 | 119 | 1.6 | 57 |\n| jcb | 55 | 61 | 0.6 | 58 |\n| diners club | 29 | 30 | 0.2 | 7 |\nquestion:\nwhat is the average payment volume per transaction for american express?",
                "rationale": "To find the average payment volume per transaction for American Express, we need to divide the total payment volume by the total number of transactions. From the table, we can see that the payment volume for American Express is $637 billion and the total number of transactions is 5.0 billion. To find the average payment volume per transaction, we divide $637 billion by 5.0 billion: Average payment volume per transaction = $637 billion / 5.0 billion = $127.4.",
                "answer": "127.4"
            }
        }
        demo = demo_dict[data]
        return demo

    def create_demo_input(
            self,
            demos: List[dict],
            demo_prefix: str,
            cot_trigger: str,
            answer_trigger: str
    ) -> Union[str, List[Dict[str, str]]]:
        if "davinci" in self.model_type:
            demo_input = demo_prefix + "\n"
            for demo in demos:
                demo_input += ("Q: " + demo["question"] + "\n" +
                               "A: " + cot_trigger + demo["rationale"] +
                               answer_trigger + demo["answer"] + "\n\n")
            return demo_input
        elif "turbo" in self.model_type:
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

    def try_table_qa_kit_by_id(
            self,
            data: str,
            dataset_type: str,
            index: int,
            question: str,
            temperature: float = 0.1,
            max_length: int = 256,
            api_time_interval: float = 1.0,
            cot_trigger: str = "Let's think step by step. ",
            answer_trigger: str = "Therefore, the answer to the question is ",
            demo_prefix: str = "Reading the texts and tables and try your best to answer the question."
    ):
        if data not in ["finqa", "tatqa", "wikisql", "wikitq", "spreadsheetqa"]:
            raise NotImplementedError
        if dataset_type not in ["train", "dev", "eval", "test"]:
            raise NotImplementedError

        dataset = DATA_CLASS[data](DATA_PATH[data][dataset_type], None).data
        # if data == "tatqa":
        #     temp_data = []
        #     for one in dataset:
        #         if len(temp_data) == 0 or temp_data[-1]["rows"] != one["rows"]:
        #             temp_data.append(one)
        #     dataset = temp_data
        demo = self.get_demo_by_data(data)
        demo_input = self.create_demo_input([demo], demo_prefix, cot_trigger, answer_trigger)
        self.model.set_prompt(demo_input)

        data_input = self.create_data_input(dataset, index, question, cot_trigger)
        output = self.model.getResponse(
            data_input,
            temperature,
            max_length,
            api_time_interval
        )
        if answer_trigger not in output:
            answer_extract = self.model.getResponse(
                data_input + output + answer_trigger,
                temperature,
                max_length,
                api_time_interval
            )
        else:
            answer_extract = output.split(answer_trigger)[-1]
        return answer_extract

    def try_table_qa_kit(
            self,
            table: Optional[List[List[str]]],
            texts: Optional[List[str]],
            question: str,
            temperature: float = 0.1,
            max_length: int = 256,
            api_time_interval: float = 1.0,
            cot_trigger: str = "Let's think step by step. ",
            answer_trigger: str = "Therefore, the answer to the question is ",
            demo_prefix: str = "Reading the texts and tables and try your best to answer the question."
    ) -> str:
        table_input = ""
        texts_input = ""
        if table is not None:
            table_input = "table:\n"
            for row in table:
                table_input += (self.table_markdown(row) + "\n")

        if texts is not None:
            texts_input = "paragraphs:\n"
            for text in texts:
                texts_input += (text + "\n")

        if "davinci" in self.model_type:
            data_input = "Q:\n" + texts_input + table_input + "question:\n" + question + "\nA:\n" + cot_trigger
        elif "turbo" in self.model_type:
            data_input = texts_input + table_input + "question:\n" + question + "\n" + cot_trigger
        else:
            raise NotImplementedError
        demo_input = self.create_demo_input([], demo_prefix, cot_trigger, answer_trigger)
        self.model.set_prompt(demo_input)
        output = self.model.getResponse(
            data_input,
            temperature,
            max_length,
            api_time_interval
        )
        if answer_trigger not in output:
            answer_extract = self.model.getResponse(
                data_input + output + answer_trigger,
                temperature,
                max_length,
                api_time_interval
            )
        else:
            answer_extract = output.split(answer_trigger)[-1]
        return answer_extract

    def create_data_input(self, dataset: List[Dict], index: int, question_in: str, cot_trigger: str, truncation: int = 3000) -> str:
        if index >= len(dataset):
            raise ValueError("Index must less than length of dataset")
        data = dataset[index]
        content = ""
        if data["texts"] is not None and len(data["texts"]):
            content += "paragraphs:\n"
            for text in data["texts"]:
                content += (text + "\n")
        if data["rows"] is not None and len(data["rows"]):
            content += "table:\n"
            for i, row in enumerate(data["rows"]):
                content += (self.table_markdown(row) + "\n")
        if "turbo" in self.model_type:
            question = "question:\n" + question_in + "\n" + cot_trigger
        else:
            question = "question:\n" + question_in + "\nA: " + cot_trigger
        if len(content) > truncation - len(question):
            content = content[:truncation - len(question) - 1] + '\n'
        content += question
        if "davinci" in self.model_type:
            content = "Q: \n" + content
        return content


if __name__ == "__main__":
    import time
    t1 = time.time()
    kit = TableQAKitDemo(model_type="text_davinci_003", key="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")
    t2 = time.time()
    for dataset_name in ["tatqa","wikisql","wikitq","spreadsheetqa"]:
        for split in ["train","dev","test"]:


            out = kit.try_table_qa_kit_by_id(data=dataset_name, dataset_type=split, index=1, question="Summarize the table?")
    # print(out)
    
    # table = [
    #     [
    #         "Player",
    #         "No.",
    #         "Nationality",
    #         "Position",
    #         "Years in Toronto",
    #         "School/Club Team"
    #     ],
    #     [
    #         "Aleksandar Radojevi\u0107",
    #         "25",
    #         "Serbia",
    #         "Center",
    #         "1999-2000",
    #         "Barton CC (KS)"
    #     ],
    #     [
    #         "Shawn Respert",
    #         "31",
    #         "United States",
    #         "Guard",
    #         "1997-98",
    #         "Michigan State"
    #     ],
    #     [
    #         "Quentin Richardson",
    #         "N/A",
    #         "United States",
    #         "Forward",
    #         "2013-present",
    #         "DePaul"
    #     ],
    #     [
    #         "Alvin Robertson",
    #         "7, 21",
    #         "United States",
    #         "Guard",
    #         "1995-96",
    #         "Arkansas"
    #     ],
    #     [
    #         "Carlos Rogers",
    #         "33, 34",
    #         "United States",
    #         "Forward-Center",
    #         "1995-98",
    #         "Tennessee State"
    #     ],
    #     [
    #         "Roy Rogers",
    #         "9",
    #         "United States",
    #         "Forward",
    #         "1998",
    #         "Alabama"
    #     ],
    #     [
    #         "Jalen Rose",
    #         "5",
    #         "United States",
    #         "Guard-Forward",
    #         "2003-06",
    #         "Michigan"
    #     ],
    #     [
    #         "Terrence Ross",
    #         "31",
    #         "United States",
    #         "Guard",
    #         "2012-present",
    #         "Washington"
    #     ]
    # ]
    # out = kit.try_table_qa_kit(table=table, texts=None, question="Summarize the table?")
            print(f"{dataset_name},{split}:",out)
    t3 = time.time()
    print("t2-t1",t2-t1)
    print("t3-t2",t3-t2)
