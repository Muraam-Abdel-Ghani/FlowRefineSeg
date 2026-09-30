import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TemporalRefinementDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.pairs = []

        for i in range(1, len(base_dataset)):
            # print(i)
            # print(len(base_dataset.sequence_labels))
            # ensure consecutive frames belong to the same sequence
            if base_dataset.sequence_labels[i] == base_dataset.sequence_labels[i - 1]:
                # print(len(base_dataset.sequence_labels))
                self.pairs.append((i - 1, i))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prev_idx, curr_idx = self.pairs[idx]
        prev_img, prev_mask, _ = self.base_dataset[prev_idx]
        curr_img, curr_mask, _ = self.base_dataset[curr_idx]
        return prev_img, curr_img, prev_mask, curr_mask