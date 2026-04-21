"""
ResNet18 model for NOx emissions regression from TEMPO imagery and wind data
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

class NOxResNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=False)
        backbone.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Linear(512 + 4, RESNET_HEAD_DIM),  # 512 image + wind_u + wind_v + num_adj + prev_qtr_mass
            nn.BatchNorm1d(RESNET_HEAD_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(RESNET_HEAD_DIM, 1),
        )

    def forward(self, image, wind_u, wind_v, num_adj, prev_qtr_mass):
        x = self.backbone(image).flatten(1)
        x = torch.cat([x, wind_u, wind_v, num_adj, prev_qtr_mass], dim=1)
        return self.head(x).squeeze(1)