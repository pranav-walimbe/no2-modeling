"""                      
ResNet9 model for NOx emissions regression from TEMPO imagery and wind data
""" 
    
import torch                                                                                                                
import torch.nn as nn

class ResBlock(nn.Module):                                                                                                    
    def __init__(self, channels):                                                                                             
        super().__init__()                                                                                                    
        self.block = nn.Sequential(                                                                                           
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),                                              
            nn.BatchNorm2d(channels),       
            nn.ReLU(),                                                                                                        
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),                                                                                         
            nn.ReLU()                       
        )                                                                                                                     
                
    def forward(self, x):                                                                                                     
        return x + self.block(x)
                                                                                                                            
class NOxResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(      
            # 56x56 -> 28x28            
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),                                                                                               
            nn.ReLU(),                      
            nn.MaxPool2d(2),                                                                                                  
                
            # 28x28 -> 14x14                                                                                                  
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),                                                                                               
            nn.ReLU(),                  
            nn.MaxPool2d(2),
            ResBlock(64),                                                                                                     

            # 14x14 -> 7x7                                                                                                    
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),                                                                                                        
            nn.MaxPool2d(2),
            ResBlock(128),                                                                                                    
                                        
            nn.AdaptiveAvgPool2d(1)
        )                                                                                                                     

        self.head = nn.Sequential(                                                                                            
            nn.Linear(128 + 1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),                      
            nn.Dropout(0.3),            
            nn.Linear(64, 1)
        )                                                                                                                     

    def forward(self, image, wind):                                                                                           
        x = self.backbone(image).flatten(1)
        x = torch.cat([x, wind], dim=1)     
        return self.head(x).squeeze(1)