"""                                                                                                                         
Training script for NOx emissions regression model                                                                          
"""                                                                                                                         
                                                                                                                            
import sys                                                                                                                  
import os                                                                                                                   
import torch    
import torch.nn as nn
import pandas as pd
from datetime import datetime       
import numpy as np                                                                                        
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))                            
from config import *                                                                                                        
from model.dataset import NOxDataset                                                                                          
from model.resnet_model import NOxResNet                                                                                             
from model.utils import *                                                  

def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return mean loss over the dataset"""                                                            
    model.train()
    total_loss = 0.0
    for image, wind, label in loader:                                                                                      
        image, wind, label = image.to(device), wind.to(device), label.to(device)
        optimizer.zero_grad()                                                                                               
        pred = model(image, wind)
        loss = criterion(pred, label)                                                                                       
        loss.backward()
        optimizer.step()                                                                                                    
        total_loss += loss.item() * len(label)
    return total_loss / len(loader.dataset)                                                                                 
                                                                                                                            
def val_epoch(model, loader, criterion, device):
    """Run one validation epoch and return mean loss over the dataset"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():                                                                                                   
        for image, wind, label in loader:
            image, wind, label = image.to(device), wind.to(device), label.to(device)                                        
            pred = model(image, wind)
            loss = criterion(pred, label)                                                                                   
            total_loss += loss.item() * len(label)
    return total_loss / len(loader.dataset)                                                                                                                                                                                                      

def run_inference(model, loader, device):                                                                                   
    """Run model inference on a dataloader and return predictions as a numpy array"""
    model.eval()                                                                                                            
    preds = []
    with torch.no_grad():                                                                                                   
        for image, wind, _ in loader:
            image, wind = image.to(device), wind.to(device)
            preds.append(model(image, wind).cpu().numpy())                                                                  
    return np.concatenate(preds)                                                                                                                 
                
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                                      

    # create run directory                                                                                                  
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_name)                                                                              
    ckpt_dir = os.path.join(run_dir, "checkpoints")                                                                         
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)                                                                                     
                                                                                                                            
    # compute normalization stats from training set                                                                         
    stats = compute_stats(split="train")                                                                                    
                
    # initialize datasets and dataloaders                                                                                   
    train_dataset = NOxDataset("train", stats=stats)                                                                                                                         
    val_dataset = NOxDataset("val", stats=stats)                                                                                                                         
    test_dataset = NOxDataset("test", stats=stats)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)                                           
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)                                            
                
    # initialize model, optimizer, loss, scheduler                                                                          
    model = NOxResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)                                                                 
    criterion = nn.MSELoss()                                                                                            
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=SCHEDULER_PATIENCE,              
        factor=SCHEDULER_FACTOR)                                                                                                                                                                                                                
    best_val_loss = float("inf")                                                                                            
    train_losses, val_losses = [], []                                                                                       
    epochs_no_improve = 0

    # main training loop                                                                                                                  
    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)                                                          
        scheduler.step(val_loss)                                                                                                  
        train_losses.append(train_loss)
        val_losses.append(val_loss)                                                                                           
        lr = optimizer.param_groups[0]["lr"]                                                                                
        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | lr: {lr:.2e}")
                                                                                                                            
        if val_loss < best_val_loss:
            best_val_loss = val_loss                                                                                        
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,                                                                                             
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),                                                             
                "val_loss": val_loss,
                "stats": stats,
            }, os.path.join(ckpt_dir, "best_model.pt"))  

        # early stopping                                                                   
        else:                                                                                                               
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:                                                                    
                print(f"Early stopping at epoch {epoch}")                 
                break                                                     

    # generate train loss visualization                                                                                                                
    plot_loss_curve(train_losses, val_losses, run_dir)                                                                      

    # load best model checkpoint for inference
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)                                                                                          
    model.load_state_dict(ckpt["model_state_dict"])     

    # run inference on all splits and generate eval visualizations                                                                                                           
    train_df = pd.read_csv(os.path.join(DATASET_DF, "train_df.csv"))                                                                                                         
    train_df["y_true"] = train_df[LABEL_COL].values                                                                                                                          
    train_df["y_pred"] = run_inference(model, train_loader, device)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    test_df = pd.read_csv(os.path.join(DATASET_DF, "test_df.csv"))
    test_df["y_true"] = test_df[LABEL_COL].values                                                                                                                            
    test_df["y_pred"] = run_inference(model, test_loader, device)                                                                                          
    val_df = pd.read_csv(os.path.join(DATASET_DF, "val_df.csv"))                                                                                                             
    val_df["y_true"] = val_df[LABEL_COL].values
    val_df["y_pred"] = run_inference(model, val_loader, device)                                                                                                 
    generate_eval_plots(train_df, test_df, val_df, run_dir)                                                                                                                                                                                                                     

if __name__ == "__main__":                                                                                                  
    main() 