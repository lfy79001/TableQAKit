from loaders.WikiSQL import WikiSQL
from loaders.WikiTableQuestions import WikiTableQuestion
from loaders.TATQA import TATQA
from loaders.FinQA import FinQA
from loaders.HiTab import HiTab
from loaders.HybridQA import HybridQA
from loaders.MultiHiertt import MultiHiertt
from loaders.MultimodalQA import MultiModalQA
from loaders.SpreadSheetQA import SpreadSheetQA

DATASET_CLASSES = {
    "WikiSQL": WikiSQL,
    "MultimodalQA": MultiModalQA,
    "WikiTableQuestions": WikiTableQuestion,
    "TATQA": TATQA,
    "FinQA": FinQA,
    "HiTab": HiTab,
    "HybridQA": HybridQA,
    "SpreadSheetQA": SpreadSheetQA,
    "MultiHiertt": MultiHiertt
}