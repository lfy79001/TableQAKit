# TAT-QA Method
```bash
cd TAT-QA
```

## Training

### Preprocessing dataset
```bash
git clone https://huggingface.co/datasets/TableQAKit/TAT-QA ./TAT-QA/dataset_reghnt
```
### Prepare dataset

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/reg_hnt python reg_hnt/prepare_dataset.py --mode [train/dev/test]
```

Note: The result will be written into the folder `./reg_hnt/cache` default.

### Train
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python reg_hnt/trainer.py --data_dir reg_hnt/cache/ \
--save_dir ./try --batch_size 48 --eval_batch_size 1 --max_epoch 100 --warmup 0.06 --optimizer adam --learning_rate 1e-4 \
--weight_decay 5e-5 --seed 42 --gradient_accumulation_steps 12 --bert_learning_rate 1e-5 --bert_weight_decay 0.01 \
--log_per_updates 50 --eps 1e-6 --encoder roberta_large --roberta_model dataset_reghnt/roberta.large
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python reg_hnt/predictor.py --data_dir reg_hnt/cache/ \
--test_data_dir reg_hnt/cache/ --save_dir reg_hnt --eval_batch_size 1 --model_path ./try \
--encoder roberta_large --roberta_model dataset_reghnt/roberta.large --mode dev
```
```
python tatqa_eval.py --gold_path=dataset_reghnt/tatqa_dataset_dev.json --pred_path=reg_hnt/pred_result_on_dev.json
```

## Testing
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python reg_hnt/predictor.py \
--data_dir reg_hnt/cache/ --test_data_dir reg_hnt/cache/ --save_dir reg_hnt \
--eval_batch_size 1 --model_path ./try --encoder roberta_large --roberta_model dataset_reghnt/roberta.large --mode test
```

