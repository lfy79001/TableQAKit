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
Download [HybridQA_Row](https://huggingface.co/datasets/TableQAKit/HybridQA_Row).

```bash
cd Noise_free
# stage1
python retrieve.py \
--ptm_type  <BERT-base>  \
--train_data_path  <DATA_FOLDER_PATH>/train.json \
--dev_data_path  <DATA_FOLDER_PATH>/dev.json \
--predict_save_path <DATA_FOLDER_PATH>/dev.row.json \
--is_train 1 \
--is_test 0 \
--is_firststage 1 \
--output_dir retrieve1 \
--load_dir retrieve1  \
--config_file config_retrieve1.json

# stage2
python retrieve.py \
--ptm_type  <BERT-base>  \
--train_data_path  <DATA_FOLDER_PATH>/train.json \
--dev_data_path  <DATA_FOLDER_PATH>/dev.json \
--predict_save_path <DATA_FOLDER_PATH>/dev.row.json \
--is_train 1 \
--is_test 0 \
--is_firststage 0 \
--output_dir retrieve2 \
--load_dir retrieve1  \
--config_file  config_retrieve2.json
```





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

```bash
cd Noise_free
# stage1
python read.py \
--ptm_type  <BART-large>  \
--train_data_path  <DATA_FOLDER_PATH>/train.json \
--dev_data_path  <DATA_FOLDER_PATH>/dev.json \
--predict_save_path <DATA_FOLDER_PATH>/dev.row.json \
--is_train 1 \
--is_test 0 \
--is_firststage 0 \
--output_dir read1 \
--load_dir read1  \
--config_file config_read.json

```