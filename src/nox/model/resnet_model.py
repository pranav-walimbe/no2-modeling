"""
Custom ResNet for NOx emissions regression from TEMPO imagery
"""

import sys
import os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.residual(x))

class NOxModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 2x48x48 -> 64x48x48
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 64x48x48 -> 64x48x48
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        # 64x48x48 -> 128x24x24
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        # 128x24x24 -> 256x12x12
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )
        # 256x12x12 -> 256x6x6
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(512 + 4, HEAD_DIM),
            nn.BatchNorm1d(HEAD_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_DIM, 1),
        )

    def forward(self, image, num_adj, prev_qtr_mass, u10, v10):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = torch.cat([x, num_adj, prev_qtr_mass, u10, v10], dim=1)
        return self.head(x).squeeze(1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)