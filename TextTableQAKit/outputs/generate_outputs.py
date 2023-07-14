import json
import os


def generate_gold_answer(dataset, split):
    if dataset == 'finqa':
        result = []
        output_file_base_path = 'finqa/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/finqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'tatqa':
        result = []
        output_file_base_path = 'tatqa/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/tatqa'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w') as file:
            for item in result:
                file.write(str(item) + '\n')
    elif dataset == 'hitab':
        result = []
        output_file_base_path = 'hitab/gold_answer'
        output_file_path = os.path.join(output_file_base_path, f'{split}.txt')
        input_file_base_path = '../datasets/hitab'
        input_file_path = os.path.join(input_file_base_path, f'{split}.json')
        with open(input_file_path, 'r') as f:
            question_data_list = json.load(f)
        for question_data in question_data_list:
            result.append(question_data['qa']['exe_ans'])
        with open(output_file_path, 'w') as file:
            for item in result:
                file.write(str(item) + '\n')



if __name__ == '__main__':
    dataset_list = ['finqa', 'tatqa', 'hitab']
    split_list = ['train', 'dev', 'test']
    for dataset in dataset_list:
        for split in split_list:
            generate_gold_answer(dataset, split)
