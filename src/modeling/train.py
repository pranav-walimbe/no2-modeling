"""Train a mask-aware model for normalized hourly NOx-mass changes."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import MODEL_IMAGE_CLIP_Z, NUM_CORES, RUNS_DIR
from modeling.dataset import (
    MODEL_FEATURE_NAMES,
    NormalizationStats,
    NOxDataset,
    compute_stats,
    denormalize_target,
    load_stats,
    save_stats,
)
from modeling.eval_utils import NORMALIZED_PRED_COL, NORMALIZED_TRUE_COL, add_mass_change_predictions, save_results
from modeling.plot_utils import plot_loss_curve, plot_pred_vs_true, plot_residuals, plot_spatial_error
from modeling.resnet import DEFAULT_DROPOUT, DEFAULT_HEAD_DIM, NOxModel

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 300
DEFAULT_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_GRADIENT_CLIP_NORM = 5.0
DEFAULT_SCHEDULER_PATIENCE = 10
DEFAULT_SCHEDULER_FACTOR = 0.50
DEFAULT_EARLY_STOP_PATIENCE = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, NUM_CORES))
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--huber-delta", type=float, default=DEFAULT_HUBER_DELTA)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler-factor", type=float, default=DEFAULT_SCHEDULER_FACTOR)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--stats", type=Path, help="Reuse normalization_stats.json from a compatible training split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inputs", choices=("full", "image", "tabular"), default="full")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = ("batch_size", "epochs", "prefetch_factor", "head_dim")
    if any(getattr(args, name) < 1 for name in positive_integer_names):
        raise ValueError(f"These arguments must be positive: {', '.join(positive_integer_names)}")
    if args.seed < 0 or args.scheduler_patience < 0:
        raise ValueError("seed and scheduler patience cannot be negative")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.workers > NUM_CORES:
        raise ValueError(f"workers cannot exceed the allocated CPU count ({NUM_CORES})")
    if args.learning_rate <= 0 or args.huber_delta <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("learning rate, Huber delta, and gradient clip norm must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight decay cannot be negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 < args.scheduler_factor < 1:
        raise ValueError("scheduler factor must be in (0, 1)")
    if args.early_stop_patience <= args.scheduler_patience:
        raise ValueError("early-stop patience must exceed scheduler patience")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    image, tabular, target, index = batch
    non_blocking = device.type == "cuda"
    return (
        image.to(device, non_blocking=non_blocking),
        tabular.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
        index,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        image, tabular, target, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(image, tabular)
            loss = criterion(prediction, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * target.numel()
    return total_loss / len(loader.dataset)


def val_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, tabular, target, _ = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(image, tabular), target)
            total_loss += loss.item() * target.numel()
    return total_loss / len(loader.dataset)


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, tabular, _, index = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(image, tabular)
            predictions.append(prediction.float().cpu().numpy())
            indices.append(index.numpy())
    return np.concatenate(predictions), np.concatenate(indices)


def _loader(dataset: NOxDataset, *, shuffle: bool, args: argparse.Namespace, device: torch.device) -> DataLoader:
    options: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers:
        options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(dataset, **options)


def _prediction_frame(
    dataset: NOxDataset,
    predictions: np.ndarray,
    indices: np.ndarray,
    stats: NormalizationStats,
) -> pd.DataFrame:
    frame = dataset.frame.iloc[indices].copy().reset_index(drop=True)
    frame[NORMALIZED_TRUE_COL] = frame["delta_nox_norm"].to_numpy(dtype=np.float64)
    frame[NORMALIZED_PRED_COL] = denormalize_target(predictions, stats)
    return add_mass_change_predictions(frame)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = _device(args.device)

    stats = load_stats(args.stats) if args.stats else compute_stats("train")
    run_name = datetime.now(timezone.utc).strftime("delta_nox_%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    save_stats(stats, run_dir / "normalization_stats.json")
    datasets = {
        split: NOxDataset(split, stats, load_images=args.inputs != "tabular") for split in ("train", "val", "test")
    }
    train_loader = _loader(datasets["train"], shuffle=True, args=args, device=device)
    eval_loaders = {
        split: _loader(dataset, shuffle=False, args=args, device=device) for split, dataset in datasets.items()
    }

    model = NOxModel(
        n_tabular_features=len(MODEL_FEATURE_NAMES),
        use_image=args.inputs in ("full", "image"),
        use_tabular=args.inputs in ("full", "tabular"),
        head_dim=args.head_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=args.huber_delta)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_path = checkpoint_dir / "best_model.pt"
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs_without_improvement = 0

    run_config = {
        "device": str(device),
        "inputs": args.inputs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "maximum_epochs": args.epochs,
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
        "head_dim": args.head_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "huber_delta": args.huber_delta,
        "gradient_clip_norm": args.gradient_clip_norm,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "early_stop_patience": args.early_stop_patience,
        "image_clip_z": MODEL_IMAGE_CLIP_Z,
        "tabular_features": list(MODEL_FEATURE_NAMES),
        "model_parameters": model.num_params(),
    }
    with (run_dir / "run_config.json").open("w") as destination:
        json.dump(run_config, destination, indent=2)
    print(f"Training {model.num_params():,} parameters on {device}; outputs: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.gradient_clip_norm,
        )
        validation_loss = val_epoch(model, eval_loaders["val"], criterion, device)
        scheduler.step(validation_loss)
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train {train_loss:.5f} | val {validation_loss:.5f} | lr {learning_rate:.2e}")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "normalization_stats": stats.to_dict(),
                    "model_feature_names": MODEL_FEATURE_NAMES,
                    "run_config": run_config,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    plot_loss_curve(train_losses, val_losses, run_dir)
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_frames = {}
    for split, loader in eval_loaders.items():
        predictions, indices = run_inference(model, loader, device)
        split_frames[split] = _prediction_frame(datasets[split], predictions, indices, stats)

    plot_pred_vs_true(split_frames, run_dir)
    plot_residuals(split_frames, run_dir)
    plot_spatial_error(split_frames, run_dir)
    save_results(split_frames, run_dir, train_target_mean=stats.target_mean)


if __name__ == "__main__":
    main()
