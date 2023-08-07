import torch
from torch.utils.data import Dataset


class RetrieveDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class RetrieveRowDataset(Dataset):
    def __init__(self, data_list):
        self.data = []
        self.positive_index = []
        idx = 0
        for one in data_list:
            for i in range(len(one["rows"])):
                sample = {"id": one["id"], "question": one["question"], "rows": one["rows"][i], "labels": []}
                sample["labels"].append(one["labels"][i])
                self.data.append(sample)
                if one["labels"][i]:
                    self.positive_index.append(idx)
                idx = idx + 1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def make_weights(self):
        weights = torch.ones(self.__len__()).float().cuda()
        weights[self.positive_index] = (self.__len__() - len(self.positive_index)) / len(self.positive_index)
        return weights
