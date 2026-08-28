"""
Define NOxDataset class for dataloader in ML pipeline
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATASET_DF, IMAGES_DIR, LABEL_COL, MAX_IMG_VAL


class NOxDataset(Dataset):
    def __init__(self, split, stats=None):
        df = pd.read_csv(os.path.join(DATASET_DF, f"{split}_df.csv"))
        self.images = np.load(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"), mmap_mode="r")
        self.labels = df[LABEL_COL].values.astype(np.float32)
        self.num_adj = df["num_adj_units"].values.astype(np.float32)
        self.prev_qtr_mass = df["prev_qtr_mass"].values.astype(np.float32)
        self.u10 = df["u10"].values.astype(np.float32)
        self.v10 = df["v10"].values.astype(np.float32)
        self.stats = stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(np.array(self.images[idx], copy=True))
        label = torch.tensor(self.labels[idx])
        num_adj = torch.tensor([self.num_adj[idx]])
        prev_qtr_mass = torch.tensor([self.prev_qtr_mass[idx]])
        u10 = torch.tensor([self.u10[idx]])
        v10 = torch.tensor([self.v10[idx]])

        # apply ceiling on no2 concentration values
        image[0] = torch.clamp(image[0], max=MAX_IMG_VAL)

        if self.stats is not None:
            image = (image - self.stats["image_mean"]) / self.stats["image_std"]
            num_adj = (num_adj - self.stats["num_adj_mean"]) / self.stats["num_adj_std"]
            prev_qtr_mass = (prev_qtr_mass - self.stats["prev_qtr_mass_mean"]) / self.stats["prev_qtr_mass_std"]
            u10 = (u10 - self.stats["u10_mean"]) / self.stats["u10_std"]
            v10 = (v10 - self.stats["v10_mean"]) / self.stats["v10_std"]

        return image, num_adj, prev_qtr_mass, u10, v10, label
