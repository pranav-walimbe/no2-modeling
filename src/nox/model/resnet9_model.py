"""                                                                                                                           
ResNet9 model for NOx emissions regression from TEMPO imagery and wind data                                                   
"""                                                                                                                           
                                                                                                                            
import sys                                                                                                                    
import os                                                                                                                     
import torch                                                                                                                  
import torch.nn as nn                                                                                                         
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *                        
                                        
def conv_block(in_channels, out_channels, pool=False):
    layers = [                                                                                                                
        nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, bias=False),
        nn.BatchNorm2d(out_channels),                                                                                         
        nn.ReLU(inplace=True)           
    ]
    if pool:                                                                                                                  
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)                                                                                             
                                            
class NOxResNet(nn.Module):             
    def __init__(self):
        super().__init__()                                                                                                    

        # Input: 1 x 56 x 56                                                                                                  
        self.conv1 = conv_block(1, 32)                  # 32 x 56 x 56
        self.conv2 = conv_block(32, 64, pool=True)      # 64 x 28 x 28
        self.res1 = nn.Sequential(      
            conv_block(64, 64),
            conv_block(64, 64)                                                                                                
        )                                               # 64 x 28 x 28
                                                                                                                            
        self.conv3 = conv_block(64, 128, pool=True)     # 128 x 14 x 14                                                       
        self.conv4 = conv_block(128, 256, pool=True)    # 256 x 7 x 7
        self.res2 = nn.Sequential(                                                                                            
            conv_block(256, 256),       
            conv_block(256, 256)
        )                                               # 256 x 7 x 7                                                         
                                        
        self.pool = nn.Sequential(                                                                                            
            nn.AdaptiveAvgPool2d(1),                    # 256 x 1 x 1                                                         
            nn.Flatten()                                # 256
        )                                                                                                                     
                                        
        self.head = nn.Sequential(
            nn.Linear(256 + 1, RESNET_HEAD_DIM),                                                                              
            nn.BatchNorm1d(RESNET_HEAD_DIM),
            nn.ReLU(inplace=True),                                                                                            
            nn.Dropout(DROPOUT),            
            nn.Linear(RESNET_HEAD_DIM, 1)
        )
                                                                                                                            
    def forward(self, image, wind):
        out = self.conv1(image)                                                                                               
        out = self.conv2(out)
        out = self.res1(out) + out          
        out = self.conv3(out)           
        out = self.conv4(out)
        out = self.res2(out) + out                                                                                            
        out = self.pool(out)
        out = torch.cat([out, wind], dim=1)                                                                                   
        return self.head(out).squeeze(1)