import torch
from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, context_window=512):
        pass

    def __len__(self):
        len(self.text)

    def __getitem__(self, index):
        pass

    def collate_fn(self, list_of_seq: list[torch.Tensor]):
        pass
