import os
from mmhqa_icl import MMQA_Runner


runner = MMQA_Runner(api_keys=['sk-123456789'])


runner.predict()

"""
python test_mmhqa_icl.py \
--data_path /home/lfy/lwh/data/mmqa \

"""