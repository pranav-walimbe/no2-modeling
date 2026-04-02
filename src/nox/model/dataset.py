"""                                                                                                                                                                                                                                                  
Define NOxDataset class for dataloader in ML pipeline                                                                                                                                                                                               
"""                                                                                                                         
                
import sys                                                                                                                                                               
import os       
import math
import zarr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF                                                                                                                           
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import * 

class NOxDataset(Dataset):                                                                                                                                               
    def __init__(self, split, stats=None):                                                                                                                               
        df = pd.read_csv(os.path.join(DATASET_DF, f"{split}_df.csv"))                                                                                                    
        self.images = zarr.open(os.path.join(IMAGES_DIR, f"{split}_tempo.zarr"), mode="r")                                                                               
        self.labels = df[LABEL_COL].values.astype(np.float32)                                                                                                            
        self.wind = df[WIND_COLS].values.astype(np.float32)
        self.stats = stats                                                                                                                                               
                
    def __len__(self):                                                                                                                                                   
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()
        label = torch.tensor(self.labels[idx]).float()                                                                                                                       
        u, v = self.wind[idx, WIND_COLS.index("era5_u10")], self.wind[idx, WIND_COLS.index("era5_v10")]
                                                                                                                                                                                                                                                                                                     
        wind = torch.tensor([np.sqrt(u**2 + v**2)]).float() # compute wind speed                                                                                                                                                           
        image = image.clamp(max=MAX_IMG_VAL) # apply windsoring on image concentrations
                                                                                                                                                                            
        if self.stats is not None:
            image = (image - self.stats["image_mean"]) / self.stats["image_std"]                                                                                             
            wind = (wind - self.stats["wind_mean"]) / self.stats["wind_std"]
                                                                                                                                                                            
        # align wind direction across images
        angle_deg = math.degrees(math.atan2(float(v), float(u)))                                                                                                             
        image = TF.rotate(image, angle=-angle_deg, fill=0.0)
                                                                                                                                                                            
        return image, wind, label