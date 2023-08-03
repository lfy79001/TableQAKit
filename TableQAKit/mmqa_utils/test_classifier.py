import torch
from classifier_module import ClassifierTrainer
MODEL_PATH= '/home/lfy/PTM/deberta-v3-large'

"""
CUDA_VISIBLE_DEVICES=0 python test_classifier.py \
--output_dir ./output \
--bert_model /home/lfy/PTM/deberta-v3-large \
--per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
--data_path /home/lfy/lwh/data/mmqa
"""


trainer = ClassifierTrainer()
trainer.train()