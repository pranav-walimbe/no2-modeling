"""Train and evaluate masked NO2 reconstruction."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    MASKED_PRETRAINING_DF_DIR,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_INPUT_CHANNELS,
    NUM_CORES,
)

from .dataset import MASKED_IMAGE_KEYS, MaskedNO2Dataset, compute_stats, save_stats
from .eval_utils import evaluate_reconstruction, save_results
from .model import ARCHITECTURE_NAME, MaskedNO2Autoencoder, masked_l1_loss
from .plot_utils import plot_loss_curve, plot_results

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
    """Parse masked-pretraining command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, NUM_CORES))
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler-factor", type=float, default=DEFAULT_SCHEDULER_FACTOR)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def _manifest_hash(dataframe_dir: str | Path = MASKED_PRETRAINING_DF_DIR) -> str:
    # Hash all published split manifests in stable order
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        with (Path(dataframe_dir) / f"{split}_df.csv").open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _loader(
    dataset: MaskedNO2Dataset,
    *,
    shuffle: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train one epoch and return masked normalized L1."""
    model.train()
    total_absolute_error = 0.0
    total_masked_pixels = 0
    amp_enabled = device.type == "cuda"
    for image, target, loss_mask, _ in loader:
        image = image.to(device, non_blocking=amp_enabled)
        target = target.to(device, non_blocking=amp_enabled)
        loss_mask = loss_mask.to(device, non_blocking=amp_enabled)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = masked_l1_loss(model(image), target, loss_mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        masked_pixels = int(loss_mask.sum().item())
        total_absolute_error += loss.detach().item() * masked_pixels
        total_masked_pixels += masked_pixels
    return total_absolute_error / total_masked_pixels


def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Calculate masked normalized L1 for one evaluation split."""
    model.eval()
    total_absolute_error = 0.0
    total_masked_pixels = 0
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for image, target, loss_mask, _ in loader:
            image = image.to(device, non_blocking=amp_enabled)
            target = target.to(device, non_blocking=amp_enabled)
            loss_mask = loss_mask.to(device, non_blocking=amp_enabled)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = masked_l1_loss(model(image), target, loss_mask)
            masked_pixels = int(loss_mask.sum().item())
            total_absolute_error += loss.item() * masked_pixels
            total_masked_pixels += masked_pixels
    return total_absolute_error / total_masked_pixels


def fit_model(
    model: MaskedNO2Autoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, object],
) -> tuple[list[float], list[float], float]:
    """Fit the model and restore its lowest-validation-loss state."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_validation_loss = float("inf")
    train_losses: list[float] = []
    validation_losses: list[float] = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.gradient_clip_norm,
        )
        val_loss = validation_loss(model, val_loader, device)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        print(
            f"Masked reconstruction epoch {epoch:03d} | train {train_loss:.5f} | "
            f"val {val_loss:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "validation_loss": val_loss,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping masked reconstruction at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return train_losses, validation_losses, best_validation_loss


def main() -> None:
    """Train masked reconstruction and compare it with interpolation."""
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)
    manifest_hash = _manifest_hash()
    stats = compute_stats()
    run_name = datetime.now(timezone.utc).strftime("masked_no2_%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    save_stats(stats, run_dir / "normalization_stats.json")

    datasets = {split: MaskedNO2Dataset(split, stats) for split in ("train", "val", "test")}
    train_loader = _loader(datasets["train"], shuffle=True, args=args, device=device)
    eval_loaders = {
        split: _loader(dataset, shuffle=False, args=args, device=device)
        for split, dataset in datasets.items()
        if split != "train"
    }
    model = MaskedNO2Autoencoder().to(device)
    print(f"Training {model.num_params():,}-parameter masked NO2 model on {device}; outputs: {run_dir}")
    checkpoint_metadata = {
        "architecture": ARCHITECTURE_NAME,
        "input_channels": MODEL_INPUT_CHANNELS,
        "image_keys": MASKED_IMAGE_KEYS,
        "normalization_stats": stats.to_dict(),
        "training_data_manifest_sha256": manifest_hash,
    }
    train_losses, validation_losses, best_validation_loss = fit_model(
        model,
        train_loader,
        eval_loaders["val"],
        device=device,
        args=args,
        checkpoint_path=checkpoint_dir / "best_masked_no2.pt",
        checkpoint_metadata=checkpoint_metadata,
    )
    plot_loss_curve(train_losses, validation_losses, run_dir)

    results = {
        "primary_model": "masked_autoencoder",
        "comparison_model": "bilinear_interpolation",
        "splits": {
            split: evaluate_reconstruction(model, loader, stats, device) for split, loader in eval_loaders.items()
        },
    }
    save_results(results, run_dir)
    plot_results(train_losses, validation_losses, results, run_dir)
    run_config = {
        "device": str(device),
        "architecture": ARCHITECTURE_NAME,
        "objective": "masked_normalized_l1",
        "baseline": "separable_bilinear_interpolation",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "maximum_epochs": args.epochs,
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
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
        "model_parameters": model.num_params(),
        "encoder_parameters": model.encoder.num_params(),
        "best_validation_loss": best_validation_loss,
        "training_data_manifest_sha256": manifest_hash,
    }
    with (run_dir / "run_config.json").open("w") as destination:
        json.dump(run_config, destination, indent=2)


if __name__ == "__main__":
    main()
