"""                                                                                                                         
Define utility functions (normalization, plotting, loss) for ML pipeline                                                                                    
"""                                                                                                                         

import sys                                                                                                                  
import os       
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd                                                                                                         
import geopandas as gpd
import seaborn as sns
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *  

def msle_loss(pred, target, reduction="mean"):
    """MSLE loss: compares predictions in log-space to reduce sensitivity to large label values"""                            
    per_sample = (torch.log1p(pred.clamp(min=0)) - torch.log1p(target.clamp(min=0))) ** 2
    if reduction == "none":                                                                                                   
        return per_sample                   
    return per_sample.mean() 

def mse_loss(pred, target, reduction="mean"):                                                                                 
    """Mean squared error loss"""                                                                                             
    per_sample = (pred - target) ** 2                                                                                         
    if reduction == "none":                                                                                                   
        return per_sample                                                                                                     
    return per_sample.mean() 

def mae_loss(pred, target, reduction="mean"):
    """Mean absolute error loss"""
    per_sample = torch.abs(pred - target)   
    if reduction == "none":             
        return per_sample
    return per_sample.mean() 
                                                                                                    
def compute_stats(split, batch_size=512):                                                                                                                                
    """Compute mean/std normalization stats from training images and wind data"""
    images = zarr.open(os.path.join(IMAGES_DIR, f"{split}_tempo.zarr"), mode="r")                                                                                        
    df = pd.read_csv(os.path.join(DATASET_DF, f"{split}_df.csv"))
    wind = df[WIND_COLS].values.astype(np.float32)                                                                                                                       
    wind_speed = np.sqrt(wind[:, WIND_COLS.index("era5_u10")]**2 + wind[:, WIND_COLS.index("era5_v10")]**2)                                                              
    n = images.shape[0]                                                                                                                                                  
                                                                                                                                                                        
    # first pass: compute mean                                                                                                                                           
    total, sum_ = 0, 0.0
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].ravel()
        batch = batch[~np.isnan(batch)].astype(np.float64)                                                                                                               
        batch = np.clip(batch, None, MAX_IMG_VAL)
        total += len(batch)                                                                                                                                              
        sum_ += batch.sum()
    mean = sum_ / total                                                                                                                                                  

    # second pass: compute std                                                                                                                                           
    sum_sq_diff = 0.0
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].ravel()
        batch = batch[~np.isnan(batch)].astype(np.float64)
        batch = np.clip(batch, None, MAX_IMG_VAL)                                                                                                                        
        sum_sq_diff += ((batch - mean) ** 2).sum()
    std = np.sqrt(sum_sq_diff / total)                                                                                                                                   
                                                                                                                                                                        
    return {
        "image_mean": float(mean),                                                                                                                                       
        "image_std": float(std),
        "wind_mean": float(wind_speed.mean()),
        "wind_std": float(wind_speed.std()),
    }                                                                                                                                                                 
                                                                                                    
def _save(fig, run_dir, plot_name):                                                                                         
    """Save figure to run directory and close it"""
    path = os.path.join(run_dir, f"{plot_name}.png")                                                                        
    fig.savefig(path, dpi=150, bbox_inches="tight")                                                                         
    plt.close(fig)                                                                                               
                                                                                                                                      
def _plant_metrics(df):
    """Compute per-plant RMSE, MAE, MAPE, and mean true emissions"""                                                                                      
    return (                                                                                                                                              
        df.groupby(["facilityId", "facilityName", "lon", "lat"])
        .apply(lambda g: pd.Series({                                                                                                                      
            "rmse": np.sqrt(((g["y_pred"] - g["y_true"]) ** 2).mean()),
            "mae": (g["y_pred"] - g["y_true"]).abs().mean(),                                                                                              
            "mape": ((g["y_pred"] - g["y_true"]).abs() / g["y_true"].replace(0, np.nan)).mean() * 100,                                                    
            "mean_y_true": g["y_true"].mean(),                                                                                                            
        })).reset_index()                                                                                                                                 
    )            

def plot_loss_curve(train_losses, val_losses, run_dir):                                                                     
    """Plot training and validation loss across epochs"""                                                                   
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.1)                                                                        
    fig, ax = plt.subplots(figsize=(8, 5))                                                                                  
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="#4C9BE8", linewidth=2)                                         
    ax.plot(epochs, val_losses, label="Val loss", color="#E85D5D", linewidth=2)
    ax.set_xlabel("Epoch", labelpad=8)                                                                                      
    ax.set_ylabel("MAE Loss", labelpad=8)
    ax.set_title(f"Training & Validation Loss\n({run_name})", fontweight="bold", pad=14)                                    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) 
    ax.legend(fontsize=10, framealpha=0.9)                                                                                  
    plt.tight_layout()
    _save(fig, run_dir, "loss_curve")                                                                                                                                                                                                            

def plot_pred_vs_true(train_df, test_df, val_df, run_dir):
    """Scatter plot of predicted vs true NOx emissions on log scale for all splits"""
    run_name = os.path.basename(run_dir)                                                                                                                                 
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))                                                                                                                      
    for ax, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):                                                            
        ax.scatter(df["y_true"], df["y_pred"], alpha=0.4, s=8, linewidth=0, color="#4C9BE8")                                                                                    
        lims = [min(df["y_true"].min(), df["y_pred"].min()),                                                                                                                     
                max(df["y_true"].max(), df["y_pred"].max())]
        ax.plot(lims, lims, color="#222222", linewidth=1.2, linestyle="--", label="Perfect prediction")                                                                          
        ax.set_xlim(lims)                                                                                                                                                        
        ax.set_ylim(lims)
        ax.set_xlabel(f"y_true ({LABEL_COL})", labelpad=8)                                                                                                           
        ax.set_ylabel(f"y_pred ({LABEL_COL})", labelpad=8)                                                                                                           
        ax.set_title(f"{split} set", fontweight="bold", pad=14)
        ax.legend(fontsize=10, framealpha=0.9)                                                                                                                              
    fig.suptitle(f"Predicted vs. True Emissions — {run_name}", fontweight="bold", fontsize=14)                                                                           
    plt.tight_layout()                                                                                                                                                   
    _save(fig, run_dir, "pred_vs_true")
                                                                                                                                                                         
def plot_residuals(train_df, test_df, val_df, run_dir):                                                                                                                  
    """Scatter plot of absolute residuals vs true NOx emissions on log-log scale for all splits"""
    run_name = os.path.basename(run_dir)
    sns.set_theme(style="whitegrid", font_scale=1.1)                                                                                                                     
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))                                                                                                                      
    for ax, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):                                                                         
        df = df.copy()                                                                                                                                                   
        df["abs_residual"] = (df["y_pred"] - df["y_true"]).abs()                                                                                                         
        ax.scatter(df["y_true"], df["abs_residual"], alpha=0.4, s=8, linewidth=0, color="#4C9BE8")                                                                                                                                                                                                                   
        ax.set_xlabel(f"y_true  ({LABEL_COL})", labelpad=8)                                                                                                  
        ax.set_ylabel(f"|y_pred - y_true|  ({LABEL_COL})", labelpad=8)
        ax.set_title(f"{split} set", fontweight="bold", pad=14)                                                                                                          
    fig.suptitle(f"Absolute Residuals vs. True Emissions — {run_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()                                                                                                                                                   
    _save(fig, run_dir, "residuals")                                                                                                                                                                       
                
def plot_spatial_error(train_df, test_df, val_df, run_dir):
    """Heatmap of per-plant MAE overlaid on a US map for all splits"""
    run_name = os.path.basename(run_dir)                                                                                                                                 
                                                                                                                                                                        
    # get US outline for geographic visualization                                                                                                                        
    us = gpd.read_file(COUNTRIES_URL)                                                                                                                                    
    us = us[us.NAME == "United States of America"]

    sns.set_theme(style="white", font_scale=1.1)                                                                                                                         
    fig, axes = plt.subplots(3, 1, figsize=(12, 21))
    for ax, (df, split) in zip(axes, [(train_df, "train"), (test_df, "test"), (val_df, "val")]):                                                                         
        metrics = _plant_metrics(df)                                                                                                                                     
        us.plot(ax=ax, color="lightgray", edgecolor="black")
        sc = ax.scatter(                                                                                                                                                 
            metrics["lon"], metrics["lat"],
            c=metrics["mae"], cmap="YlOrRd", s=40, alpha=0.85,
            edgecolors="black", linewidths=0.3,                                                                                                                          
            vmin=metrics["mae"].min(), vmax=metrics["mae"].max()
        )                                                                                                                                                                
        plt.colorbar(sc, ax=ax, label=f"MAE  ({LABEL_COL})")
        ax.set_xlim(-130, -65)                                                                                                                                           
        ax.set_ylim(24, 50)
        ax.set_xlabel("Longitude", labelpad=8)                                                                                                                           
        ax.set_ylabel("Latitude", labelpad=8)                                                                                                                            
        ax.set_title(f"{split} set", fontweight="bold", pad=14)
    fig.suptitle(f"Spatial Error Distribution — {run_name}", fontweight="bold", fontsize=14)                                                                             
    plt.tight_layout()                                                                                                                                                   
    _save(fig, run_dir, "spatial_error")

def plot_residual_examples(test_df, run_dir, n=10):                                                                                                       
    """For n random plants, plot their lowest and highest residual prediction"""                                                                          
    run_name = os.path.basename(run_dir)                                                                                                                  
    df = test_df.copy().reset_index(drop=True)                                                                                                            
    df["abs_residual"] = (df["y_pred"] - df["y_true"]).abs()                                                                                              
    images = zarr.open(os.path.join(IMAGES_DIR, "test_tempo.zarr"), mode="r")                                                                             
                                                                                                                                                        
    plants = df["facilityId"].drop_duplicates().sample(n=min(n, df["facilityId"].nunique()), random_state=42)                                             
                                                                                                                                                        
    fig, axes = plt.subplots(2, len(plants), figsize=(3 * len(plants), 7))
    fig.suptitle(f"Per-plant residual examples (test set — {run_name})", fontweight="bold", fontsize=14)                                                  
    axes[0, 0].set_ylabel("Low residual", fontsize=10, labelpad=8)
    axes[1, 0].set_ylabel("High residual", fontsize=10, labelpad=8)                                                                                       
                                            
    for i, facility_id in enumerate(plants):                                                                                                              
        plant_df = df[df["facilityId"] == facility_id]
        low_row  = plant_df.loc[plant_df["abs_residual"].idxmin()]                                                                                        
        high_row = plant_df.loc[plant_df["abs_residual"].idxmax()]
                                                                                                                                                        
        for ax, row in [(axes[0, i], low_row), (axes[1, i], high_row)]:                                                                                   
            img = images[int(row["zarr_idx"])][0]
            ax.imshow(img, cmap="viridis", interpolation="nearest")                                                                                       
            ax.set_title(f"{row['facilityName']}\nresidual={row['abs_residual']:.2f}\ntrue={row['y_true']:.2f}", fontsize=6)                              
            ax.axis("off")                                                                                                                                
                                                                                                                                                        
    plt.tight_layout()
    _save(fig, run_dir, "residual_examples") 

def print_mae_summary(train_df, val_df, test_df):
    """Print MAE summary statistics across all splits""" 
    print("\n--- MAE by split ---")                                                                        
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        mae = np.abs(df["y_true"].values - df["y_pred"].values).mean()
        print(f"  {split:<6} MAE: {mae:.4f}")
                                                                                                                            
    # stratified MAE on test set by label tertile                                                                             
    print("\n--- Test MAE by emission tertile ---")                                                                         
    t33, t66 = np.percentile(test_df["y_true"], [33, 66])                                                                     
    low = test_df[test_df["y_true"] <  t33]                                                                                  
    mid = test_df[(test_df["y_true"] >= t33) & (test_df["y_true"] < t66)]
    high = test_df[test_df["y_true"] >= t66]                                                                                  
                                                                                                                            
    for name, subset in [("low", low), ("mid", mid), ("high", high)]:                                                         
        mae = np.abs(subset["y_true"].values - subset["y_pred"].values).mean()                                                
        label_range = f"[{subset['y_true'].min():.1f}, {subset['y_true'].max():.1f}]"                                                 
        print(f"  {name:<6} MAE: {mae:.4f}  (n={len(subset)}, label range {label_range})") 
                                                                                                                                                                        
def generate_eval_plots(train_df, test_df, val_df, run_dir):                                                                                                             
    """Generate all inference evaluation plots for a completed training run"""
    plot_pred_vs_true(train_df, test_df, val_df, run_dir)
    plot_residuals(train_df, test_df, val_df, run_dir)                                                                                                                   
    plot_spatial_error(train_df, test_df, val_df, run_dir)
    plot_residual_examples(test_df, run_dir) 