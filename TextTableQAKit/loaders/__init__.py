from loaders.wikisql import WikiSQL
from loaders.wikitq import WikiTableQuestion
from loaders.tatqa import TATQA
from loaders.finqa import FinQA
from loaders.hitab import HiTab
from loaders.hybridqa import HybridQA
# from loaders.multihiertt import MultiHiertt
from loaders.mmqa import MultiModalQA


DATASET_CLASSES = {
    "wikisql": WikiSQL,
    "mmqa": MultiModalQA,
    "wikitq": WikiTableQuestion,
    "tatqa": TATQA,
    "finqa": FinQA,
    "hitab": HiTab,
    "hybridqa": HybridQA,
    # "multihiertt": MultiHiertt
}