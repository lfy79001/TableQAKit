# Multi-hop TableQA

## Retrieval
Download train_data, dev_data, WikiTables from our [huggingface datasets repository](https://huggingface.co/TableQAKit)

### General Retriever
Download [HybridQA_R](https://huggingface.co/datasets/TableQAKit/HybridQA_R).

#### Row Retriever
```bash
CUDA_VISIBLE_DEVICES=5 python main_retriever.py \
--train_data_path <YOUR_FOLDER_PATH>/train.json \
--dev_data_path <YOUR_FOLDER_PATH>/dev.json \
--WikiTables <YOUR_FOLDER_PATH>WikiTables-WithLinks \
--is_train 1 \
--output_path ./try \
--plm <Your PLM Path> \
--config_file config_retriever.json \
--mode row
```

#### Column Retriever
```bash
CUDA_VISIBLE_DEVICES=5 python main_retriever.py \
--train_data_path <YOUR_FOLDER_PATH>/train.json \
--dev_data_path /home/lfy/S3HQA1/Data/HybridQA/dev.json \
--WikiTables /home/lfy/S3HQA1/Data/HybridQA/WikiTables-WithLinks \
--is_train 1 \
--output_path ./try \
--plm <Your PLM Path> \
--config_file config_retriever.json \
--mode column
```

### Noise free Retriever






## Reader

### General Reader

Download [HybridQA_read](https://huggingface.co/datasets/TableQAKit/HybridQA_read).

#### MRC-base Reader
```bash
CUDA_VISIBLE_DEVICES=5 python main_reader.py \
--train_data_path <YOUR_FOLDER_PATH>/train.read.json \
--dev_data_path <YOUR_FOLDER_PATH>/dev.read.json \
--dev_reference  <YOUR_FOLDER_PATH>/dev_reference.json \
--WikiTables <YOUR_FOLDER_PATH>WikiTables-WithLinks \
--is_train 1 \
--output_path ./try \
--plm <BERT/RoBERTa/DeBERTa> \
--config_file config_reader.json
--mode mrc
```

#### Generative-base Reader
```bash
CUDA_VISIBLE_DEVICES=5 python main_reader.py \
--train_data_path <YOUR_FOLDER_PATH>/train.read.json \
--dev_data_path <YOUR_FOLDER_PATH>/dev.read.json \
--dev_reference  <YOUR_FOLDER_PATH>/dev_reference.json \
--WikiTables <YOUR_FOLDER_PATH>WikiTables-WithLinks \
--is_train 1 \
--output_path ./try \
--plm <BART> \
--config_file config_reader.json
--mode generate
```

### Noise free Reader
Download [HybridQA_Row](https://huggingface.co/datasets/TableQAKit/HybridQA_Row).


