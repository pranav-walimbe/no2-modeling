"""Train a mask-aware binary classifier for hourly NOx-mass changes."""

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

from config import (
    DATASET_DF,
    LABEL_COL,
    MODEL_IMAGE_CLIP_ABS,
    NUM_CORES,
    RUNS_DIR,
    STRAT_BASE_DIR,
)
from modeling.dataset import (
    LABEL_MODE_COL,
    MODEL_FEATURE_NAMES,
    NOxDataset,
    clipped_pixel_fractions,
    compute_stats,
    save_stats,
)
from modeling.eval_utils import (
    LOGIT_COL,
    POSITIVE_PROBABILITY_COL,
    PREDICTED_CLASS_COL,
    TRUE_CLASS_COL,
    save_results,
)
from modeling.plot_utils import (
    plot_class_probabilities,
    plot_loss_curve,
    plot_model_comparison,
    plot_spatial_accuracy,
)
from modeling.resnet import DEFAULT_DROPOUT, DEFAULT_HEAD_DIM, NOxModel
from modeling.xgboost import train_xgboost_baseline

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 300
DEFAULT_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_NORM = 5.0
DEFAULT_SCHEDULER_PATIENCE = 10
DEFAULT_SCHEDULER_FACTOR = 0.50
DEFAULT_EARLY_STOP_PATIENCE = 25


def parse_args() -> argparse.Namespace:
    """Parse model-training command-line options.

    Returns:
        Parsed command-line arguments.
    """
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
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler-factor", type=float, default=DEFAULT_SCHEDULER_FACTOR)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inputs", choices=("full", "image", "tabular"), default="full")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    # Resolve the requested training device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    # Seed CPU and CUDA random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    # Move model inputs and targets onto the training device
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
    """Train the model for one epoch.

    Args:
        model: Model to optimize.
        loader: Training batches.
        optimizer: Parameter optimizer.
        criterion: Training loss.
        scaler: Mixed-precision gradient scaler.
        device: Training device.
        gradient_clip_norm: Maximum gradient norm.

    Returns:
        Mean training loss per record.
    """
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
    """Calculate validation loss for one epoch.

    Args:
        model: Model to evaluate.
        loader: Validation batches.
        criterion: Validation loss.
        device: Evaluation device.

    Returns:
        Mean validation loss per record.
    """
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
    """Generate logits for one dataset split.

    Args:
        model: Trained model.
        loader: Evaluation batches.
        device: Inference device.

    Returns:
        Logits and corresponding dataset indices.
    """
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
    # Configure one deterministic data loader
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
    logits: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    # Attach probabilities and thresholded classes in source-record order
    frame = dataset.frame.iloc[indices].copy().reset_index(drop=True)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    frame[TRUE_CLASS_COL] = frame[LABEL_COL].to_numpy(dtype=np.uint8)
    frame[LOGIT_COL] = logits
    frame[POSITIVE_PROBABILITY_COL] = probability
    frame[PREDICTED_CLASS_COL] = (probability >= 0.5).astype(np.uint8)
    return frame


def _load_classification_summaries(
    dataframe_dir: str | Path = DATASET_DF,
) -> dict[str, object]:
    # Preserve natural prevalence beside metrics from balanced splits
    stratification_path = Path(STRAT_BASE_DIR) / "classification_summary.json"
    with stratification_path.open() as source:
        stratification = json.load(source)

    generated = {}
    for split in ("train", "val", "test"):
        path = Path(dataframe_dir) / f"{split}_classification_summary.json"
        with path.open() as source:
            summary = json.load(source)
        generated[split] = summary
    return {"stratification": stratification, "generated_splits": generated}


def main() -> None:
    """Train and evaluate one binary-classification run."""
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)

    stats = compute_stats("train")
    classification_summaries = _load_classification_summaries()
    run_name = datetime.now(timezone.utc).strftime("delta_nox_classification_%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    save_stats(stats, run_dir / "normalization_stats.json")
    load_images = args.inputs != "tabular"
    datasets = {split: NOxDataset(split, stats, load_images=load_images) for split in ("train", "val", "test")}
    clipped_fractions = {split: clipped_pixel_fractions(split, stats) for split in datasets} if load_images else {}
    target_label_mode = str(datasets["train"].frame[LABEL_MODE_COL].iloc[0])
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
    train_criterion = nn.BCEWithLogitsLoss()
    eval_criterion = nn.BCEWithLogitsLoss()
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
        "gradient_clip_norm": args.gradient_clip_norm,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "early_stop_patience": args.early_stop_patience,
        "image_keys": list(stats.image_keys),
        "image_center": list(stats.image_center),
        "image_scale": list(stats.image_scale),
        "image_clip_range": [-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS],
        "clipped_valid_pixel_fraction": clipped_fractions,
        "raw_delta_nox_threshold": stats.delta_threshold,
        "target_label_mode": target_label_mode,
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
            train_criterion,
            scaler,
            device,
            args.gradient_clip_norm,
        )
        validation_loss = val_epoch(model, eval_loaders["val"], eval_criterion, device)
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
        logits, indices = run_inference(model, loader, device)
        split_frames[split] = _prediction_frame(datasets[split], logits, indices)

    plot_class_probabilities(split_frames, run_dir)
    plot_spatial_accuracy(split_frames, run_dir)
    print("Training XGBoost tabular baseline")
    xgboost_run = train_xgboost_baseline(
        datasets,
        checkpoint_dir / "xgboost_model.json",
        seed=args.seed,
        workers=args.workers,
    )
    run_config["xgboost"] = {
        **xgboost_run.config.to_dict(),
        "best_iteration": xgboost_run.best_iteration,
        "best_validation_logloss": xgboost_run.best_validation_logloss,
    }
    with (run_dir / "run_config.json").open("w") as destination:
        json.dump(run_config, destination, indent=2)
    model_frames = {"deep_learning": split_frames, "xgboost": xgboost_run.split_frames}
    plot_model_comparison(model_frames, run_dir)
    save_results(model_frames, classification_summaries, run_dir)


if __name__ == "__main__":
    main()
