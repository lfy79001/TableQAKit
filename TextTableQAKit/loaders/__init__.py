from .wikisql import WikiSQL
from .wikitq import WikiTableQuestion
from .tatqa import TATQA
from .finqa import FinQA
from .hitab import HiTab
from .hybridqa import HybridQA
# from .multihiertt import MultiHiertt
from .mmqa import MultiModalQA


DATASET_CLASSES = {
    "wikisql": WikiSQL,
    "multimodalqa": MultiModalQA,
    "wikitq": WikiTableQuestion,
    "tatqa": TATQA,
    "finqa": FinQA,
    "hitab": HiTab,
    "hybridqa": HybridQA,
    # "multihiertt": MultiHiertt
}