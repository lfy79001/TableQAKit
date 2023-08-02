import torch
from retriever_module import RetrieverTrainer

"""

CUDA_VISIBLE_DEVICES=0,1,2 python test_retriever.py \
--output_dir ./output \
--bert_model /home/lfy/PTM/deberta-v3-large \
--data_path /home/lfy/lwh/data/mmqa \
--n_gpu 3 \
--image_or_text image \
--data_path /home/lfy/lwh/data/mmqa
"""

trainer = RetrieverTrainer()
trainer.train()