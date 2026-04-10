"""                      
ResNet18 model for NOx emissions regression from TEMPO imagery and wind data
"""                                                                                                                         

import sys                                                                                                                  
import os       
import torch                                                                                                                
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *                                                                                                        
                                                                                                                     
class NOxResNet(nn.Module):                                                                                                                               
    def __init__(self):                                                                                                                                   
        super().__init__()
                                                                                                                                                        
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None)
        if USE_PRETRAINED:              
            pretrained_weight = backbone.conv1.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=False)                                            
        if USE_PRETRAINED:                  
            backbone.conv1.weight = nn.Parameter(pretrained_weight)                                                                                       
        backbone.maxpool = nn.Identity()    
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])                                                                                    

        self.head = nn.Sequential(                                                                                                                        
            nn.Linear(513, RESNET_HEAD_DIM),
            nn.BatchNorm1d(RESNET_HEAD_DIM),                                                                                                              
            nn.ReLU(),                  
            nn.Linear(RESNET_HEAD_DIM, 1)
        )                                                                                                                                                

    def forward(self, image, wind):                                                                                                                       
        x = self.backbone(image).flatten(1)
        x = torch.cat([x, wind], dim=1)     
        return self.head(x).squeeze(1)  