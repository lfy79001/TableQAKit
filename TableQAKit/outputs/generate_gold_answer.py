import json
import os


def generate_gold_answer(dataset, split):
    if dataset == 'SpreadSheetQA':
        result = []
        output_file_base_path = 'SpreadSheetQA/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/spreadsheetqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'FinQA':
        result = []
        output_file_base_path = 'FinQA/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/finqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            if 'exe_ans' in question_data['qa']:
                result.append(question_data['qa']['exe_ans'])
            else:
                result.append("<The test dataset does not provide a gold answer.>")
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'TAT-QA':
        result = []
        output_file_base_path = 'TAT-QA/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/tatqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            for single_question_data in question_data["questions"]:
                if 'answer' in single_question_data:
                    result.append(str(single_question_data['answer']))
                else:
                    result.append("<The test dataset does not provide a gold answer.>")
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'HiTab':
        

        result = []
        output_file_base_path = 'HiTab/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/hitab'
        input_file_path = os.path.join(input_file_base_path, f'{split}.jsonl')
        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                answer = str(json_data['answer'])
                result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')





    elif dataset == 'WikiSQL':
        result = []
        output_file_base_path = 'WikiSQL/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/wikisql'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            # if len(question_data['answer_text']) == 1:
            #     answer = question_data['answer_text'][0]
            # else:
            #     answer = str(question_data['answer_text'])
            answer = question_data['seq_out']
            result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'WikiTableQuestions':
        result = []
        output_file_base_path = 'WikiTableQuestions/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/wikitq'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            # if len(question_data['answer_text']) == 1:
            #     answer = question_data['answer_text'][0]
            # else:
            #     answer = str(question_data['answer_text'])
            answer = question_data['seq_out']
            result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == "HybridQA":
        result = []
        output_file_base_path = 'HybridQA/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/hybridqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            if 'answer-text' in question_data:
                answer = question_data['answer-text']
            else:
                answer = "<The test dataset does not provide a gold answer.>"
            result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'MultimodalQA':
        result = []
        output_file_base_path = 'MultimodalQA/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/mmqa'
        input_file_path = os.path.join(input_file_base_path, f'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_{split}.jsonl')
        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                if 'answers' in json_data:
                    answers = json_data['answers']
                    answer = str([answer_data['answer'] for answer_data in answers])
                else:
                    answer = "<The test dataset does not provide a gold answer.>"
                result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'MultiHiertt':
        result = []
        output_file_base_path = 'MultiHiertt/Gold_Answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/multihiertt'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            if 'answer' in question_data['qa']:
                answer = question_data['qa']['answer']
            else:
                answer = "<The test dataset does not provide a gold answer.>"
            result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')

if __name__ == '__main__':
    dataset_list = ['SpreadSheetQA', 'WikiSQL', 'WikiTableQuestions', 'HybridQA', 'MultimodalQA', 'TAT-QA', 'FinQA', 'HiTAB', 'MultiHiertt']
    split_list = ['train', 'dev', 'test']

    for dataset in dataset_list:
        for split in split_list:
            generate_gold_answer(dataset, split)
