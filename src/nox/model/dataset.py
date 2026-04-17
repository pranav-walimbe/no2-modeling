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
        self.wind = df["wind_speed"].values.astype(np.float32)
        self.num_adj = df["num_adj_units"].values.astype(np.float32)
        self.freq_weights = df["freq_weight"].values.astype(np.float32)                                                          
        self.stats = stats                                             
                                                                                                                                
    def __len__(self):                  
        return len(self.labels)                                                                                                  
                                                                                                                                
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()                                                                           
        label = torch.tensor(self.labels[idx]).float()
        freq_weight = torch.tensor(self.freq_weights[idx]).float()
        wind = torch.tensor([self.wind[idx]]).float()             
        num_adj = torch.tensor([self.num_adj[idx]]).float()
                                                                                                                                
        image = image.clamp(max=MAX_IMG_VAL)
                                                                                                                                
        if self.stats is not None:
            image = (image - self.stats["image_mean"]) / self.stats["image_std"]                                                 
            wind = (wind - self.stats["wind_mean"]) / self.stats["wind_std"]    
            num_adj = (num_adj - self.stats["num_adj_mean"]) / self.stats["num_adj_std"]                                         
                                        
        return image, wind, num_adj, label, freq_weight