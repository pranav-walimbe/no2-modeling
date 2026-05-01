"""
Custom DenseNet for NOx emissions regression from TEMPO imagery
"""

import sys
import os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layer(x)

class NOxModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 2x48x48 -> 64x48x48
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 64x48x48 -> 244x48x48  (64 + 6*30)
        self.block1 = DenseBlock(64, 6, 30)
        # 244x48x48 -> 122x24x24
        self.trans1 = Transition(244, 122)

        # 122x24x24 -> 302x24x24  (122 + 6*30)
        self.block2 = DenseBlock(122, 6, 30)
        # 302x24x24 -> 152x12x12
        self.trans2 = Transition(302, 152)

        # 152x12x12 -> 332x12x12  (152 + 6*30)
        self.block3 = DenseBlock(152, 6, 30)
        # 332x12x12 -> 332x1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 332 image features + 4 scalars
        self.head = nn.Sequential(
            nn.Linear(332 + 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, image, num_adj, prev_qtr_mass, u10, v10):
        x = self.stem(image)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        x = torch.cat([x, num_adj, prev_qtr_mass, u10, v10], dim=1)
        return self.head(x).squeeze(1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)