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

        # resnet18 backbone modified for single-channel input                                 
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=False)              
        backbone.maxpool = nn.Identity() # avoid aggressive pooling / convolution on small input images to preserve resolution                                                                                  
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])                                                      
                
        # regression head: image features + wind features -> scalar                                                    
        self.head = nn.Sequential(                                                                                                                                               
            nn.Linear(513, RESNET_HEAD_DIM), # RESNET_HEAD_DIM + len(wind_scalar)
            nn.BatchNorm1d(RESNET_HEAD_DIM),                                                                                                                                     
            nn.ReLU(),                      
            nn.Dropout(p=DROPOUT),
            nn.Linear(RESNET_HEAD_DIM, 1)
        )   
                                                                                                                            
    def forward(self, image, wind):
        x = self.backbone(image).flatten(1) # (B, 512)
        x = torch.cat([x, wind], dim=1) # (B, 512 + wind_dim)                                                               
        return self.head(x).squeeze(1) # (B,) 