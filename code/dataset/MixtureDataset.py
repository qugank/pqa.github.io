import random
from functools import reduce
from typing import List

from torch.utils.data import DataLoader, Dataset


class MixtureDataset(Dataset):

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.datasets_len = [len(d) for d in datasets]

    def __len__(self):
        return sum(self.datasets_len)

    def __getitem__(self, raw_index):
        index = raw_index
        for i, dataset_len in enumerate(self.datasets_len):
            if index < dataset_len:
                dataset = self.datasets[i]
                return dataset[index]
            else:
                index = index - dataset_len
