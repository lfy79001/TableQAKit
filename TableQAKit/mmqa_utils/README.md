## MMQA utils



### 分类器

使用 `ClassifierTrainer` 

```python
CUDA_VISIBLE_DEVICES=0 python test_classifier.py \
--output_dir ./output \
--bert_model /home/lfy/PTM/deberta-v3-large \
--per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
--data_path /home/lfy/lwh/data/mmqa
```





### 检索器

使用 `RetrieverTrainer` 

```python
CUDA_VISIBLE_DEVICES=0,1,2 python test_retriever.py \
--output_dir ./output \
--bert_model /home/lfy/PTM/deberta-v3-large \
--data_path /home/lfy/lwh/data/mmqa \
--n_gpu 3 \
--image_or_text image \
--data_path /home/lfy/lwh/data/mmqa
```

