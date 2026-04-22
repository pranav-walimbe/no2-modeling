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
        self.images = np.load(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"), mmap_mode="r")
        self.labels = df[LABEL_COL].values.astype(np.float32)
        self.wind_u = df[WIND_COLS[0]].values.astype(np.float32)
        self.wind_v = df[WIND_COLS[1]].values.astype(np.float32)
        self.num_adj = df["num_adj_units"].values.astype(np.float32)
        self.prev_qtr_mass = df["prev_qtr_mass"].values.astype(np.float32)
        self.stats = stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()  # (2, H, W)
        label = torch.tensor(self.labels[idx]).float()
        wind_u = torch.tensor([self.wind_u[idx]]).float()
        wind_v = torch.tensor([self.wind_v[idx]]).float()
        num_adj = torch.tensor([self.num_adj[idx]]).float()
        prev_qtr_mass = torch.tensor([self.prev_qtr_mass[idx]]).float()

        image[0] = torch.clamp(image[0], max=MAX_IMG_VAL)

        if self.stats is not None:
            image = (image - self.stats["image_mean"]) / self.stats["image_std"]
            wind_u = (wind_u - self.stats["wind_u_mean"]) / self.stats["wind_u_std"]
            wind_v = (wind_v - self.stats["wind_v_mean"]) / self.stats["wind_v_std"]
            num_adj = (num_adj - self.stats["num_adj_mean"]) / self.stats["num_adj_std"]
            prev_qtr_mass = (prev_qtr_mass - self.stats["prev_qtr_mass_mean"]) / self.stats["prev_qtr_mass_std"]

        return image, wind_u, wind_v, num_adj, prev_qtr_mass, label