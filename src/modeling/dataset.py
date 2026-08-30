"""
Define NOxDataset class for dataloader in ML pipeline
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATASET_DF, IMAGES_DIR, LABEL_COL, MAX_IMG_VAL


def compute_stats(split: str, batch_size: int = 512) -> dict:
    """Compute normalization statistics for images and tabular features."""
    images = np.load(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"))
    df = pd.read_csv(os.path.join(DATASET_DF, f"{split}_df.csv"))
    num_adj = df["num_adj_units"].values.astype(np.float32)
    prev_qtr_mass = df["prev_qtr_mass"].values.astype(np.float32)
    u10 = df["u10"].values.astype(np.float32)
    v10 = df["v10"].values.astype(np.float32)
    n, n_channels = images.shape[0], images.shape[1]

    total = 0
    sums = np.zeros(n_channels, dtype=np.float64)
    for index in range(0, n, batch_size):
        batch = images[index : index + batch_size].astype(np.float64)
        batch[:, 0] = np.clip(batch[:, 0], None, MAX_IMG_VAL)
        total += batch.shape[0] * batch.shape[2] * batch.shape[3]
        sums += batch.sum(axis=(0, 2, 3))
    means = sums / total

    sum_sq_diffs = np.zeros(n_channels, dtype=np.float64)
    for index in range(0, n, batch_size):
        batch = images[index : index + batch_size].astype(np.float64)
        batch[:, 0] = np.clip(batch[:, 0], None, MAX_IMG_VAL)
        sum_sq_diffs += ((batch - means[np.newaxis, :, np.newaxis, np.newaxis]) ** 2).sum(axis=(0, 2, 3))
    stds = np.sqrt(sum_sq_diffs / total)

    return {
        "image_mean": torch.tensor(means, dtype=torch.float32).view(n_channels, 1, 1),
        "image_std": torch.tensor(stds, dtype=torch.float32).view(n_channels, 1, 1),
        "num_adj_mean": float(num_adj.mean()),
        "num_adj_std": float(num_adj.std()),
        "prev_qtr_mass_mean": float(prev_qtr_mass.mean()),
        "prev_qtr_mass_std": float(prev_qtr_mass.std()),
        "u10_mean": float(u10.mean()),
        "u10_std": float(u10.std()),
        "v10_mean": float(v10.mean()),
        "v10_std": float(v10.std()),
    }


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
