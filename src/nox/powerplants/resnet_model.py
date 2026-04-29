"""                                                                                                                                             
Define ResNetRegressor class for NOx emission prediction in ML pipeline                                                                                   
"""                                                                                                                                             
                                                                                                                                                
import sys                                                                                                                                      
import os                                                                                                                                       
import torch                                                                                                                                  
import torch.nn as nn                                                                                                                           
from torchvision.models import resnet18                                                                                                       
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

class ResNetRegressor(nn.Module):
    """ResNet18 backbone with late fusion of wind features for NOx regression."""
    def __init__(self, n_wind_features=len(WIND_COLS)):
        super().__init__()

        backbone = resnet18(weights=None)
        # modify stem for small single-channel inputs
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        # remove classification head, keep feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # late fusion: concatenate wind features with image features
        self.head = nn.Sequential(
            nn.Linear(512 + n_wind_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image, wind):
        features = self.backbone(image).flatten(1)
        fused = torch.cat([features, wind], dim=1)
        return self.head(fused).squeeze(1)