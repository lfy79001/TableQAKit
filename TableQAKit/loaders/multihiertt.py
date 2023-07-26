import os
import platform

import re
import logging
import json
from structs.data import Cell, Table, HFTabularDataset


logger = logging.getLogger(__name__)

class MultiHiertt(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_split(self, split):
        pass

    def load_split_test(self, split):
        self._load_split(split)

    def prepare_table(self, entry):
        t = Table()
        
        return t
if __name__ == '__main__':
    MultiHiertt().load_split_test("train")