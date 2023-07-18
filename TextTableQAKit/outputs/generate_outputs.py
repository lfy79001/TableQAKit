import json
import os


def generate_gold_answer(dataset, split):
    if dataset == 'finqa':
        result = []
        output_file_base_path = 'finqa/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/finqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'tatqa':
        result = []
        output_file_base_path = 'tatqa/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/tatqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'hitab':
        result = []
        output_file_base_path = 'hitab/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/hitab'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'wikisql':
        result = []
        output_file_base_path = 'wikisql/gold_answer'
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
    elif dataset == 'wikitq':
        result = []
        output_file_base_path = 'wikitq/gold_answer'
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
    elif dataset == "hybridqa":
        result = []
        output_file_base_path = 'hybridqa/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/hybridqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r', encoding='utf-8') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            if 'answer-text' in question_data:
                answer = question_data['answer-text']
            else:
                answer = "[The test dataset does not provide a gold answer.]"
            result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'mmqa':
        result = []
        output_file_base_path = 'mmqa/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/mmqa'
        input_file_path = os.path.join(input_file_base_path, f'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_{split}.jsonl')
        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                if 'answers' in json_data:
                    answers = json_data['answers']
                    if len(answers) == 1:
                        answer = str(answers[0]['answer'])
                    else:
                        answer = str([answer_data['answer'] for answer_data in answers])
                else:
                    answer = "[The test dataset does not provide a gold answer.]"
                result.append(answer)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for item in result:
                file.write(str(item) + '\n')

if __name__ == '__main__':
    dataset_list = ['finqa', 'tatqa', 'hitab', 'wikisql', 'wikitq', 'hybridqa', 'mmqa']
    split_list = ['train', 'dev', 'test']

    for dataset in dataset_list:
        for split in split_list:
            generate_gold_answer(dataset, split)
