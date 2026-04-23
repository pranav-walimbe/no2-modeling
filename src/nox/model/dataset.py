"""
Define NOxDataset class for dataloader in ML pipeline
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import *

class NOxDataset(Dataset):
    def __init__(self, split, stats=None):
        df = pd.read_csv(os.path.join(DATASET_DF, f"{split}_df.csv"))
        self.images = np.load(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"))
        self.labels = df[LABEL_COL].values.astype(np.float32)
        self.num_adj = df["num_adj_units"].values.astype(np.float32)
        self.prev_qtr_mass = df["prev_qtr_mass"].values.astype(np.float32)
        self.stats = stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()  # (4, H, W): [no2, delta_no2, u10, v10]
        label = torch.tensor(self.labels[idx]).float()
        num_adj = torch.tensor([self.num_adj[idx]]).float()
        prev_qtr_mass = torch.tensor([self.prev_qtr_mass[idx]]).float()

        # apply ceiling on no2 concentration values
        image[0] = torch.clamp(image[0], max=MAX_IMG_VAL)

        if self.stats is not None:
            image = (image - self.stats["image_mean"]) / self.stats["image_std"]
            num_adj = (num_adj - self.stats["num_adj_mean"]) / self.stats["num_adj_std"]
            prev_qtr_mass = (prev_qtr_mass - self.stats["prev_qtr_mass_mean"]) / self.stats["prev_qtr_mass_std"]

        return image, num_adj, prev_qtr_mass, label