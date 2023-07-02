from ..structs.data import Cell, Table, HFTabularDataset

class TATQA(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        