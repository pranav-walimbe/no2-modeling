"""Train convolutional recurrent and tabular emissions-change classifiers."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from modeling.convgru import (
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_DIM,
    ENCODER_ARCHITECTURE_NAME,
    ENCODER_OUTPUT_CHANNELS,
    RasterConvGRUClassifier,
    RasterFrameEncoder,
    ResidualBlock,
)
from modeling.dataset import (
    LABEL_MODE_COL,
    MODEL_FEATURE_NAMES,
    NOxDataset,
    compute_stats,
    save_stats,
)
from modeling.eval_utils import (
    LOGIT_COLUMNS,
    PREDICTED_CLASS_COL,
    PROBABILITY_COLUMNS,
    TRUE_CLASS_COL,
    save_results,
)
from modeling.mlp import TabularMLP
from modeling.plot_utils import plot_confusion_matrices, plot_loss_curve, plot_training_comparison
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import (
    MODEL_CLASS_NAMES,
    MODEL_IMAGE_CHANNELS,
    MODEL_IMAGE_CLIP_ABS,
    MODEL_IMAGE_KEYS,
    MODEL_TARGET_COL,
    NUM_CORES,
    PRETRAINED_ENCODER_WEIGHTS,
    RUNS_DIR,
)

DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 100
DEFAULT_TABULAR_EPOCHS = 75
DEFAULT_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_TABULAR_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_NORM = 5.0
DEFAULT_SCHEDULER_PATIENCE = 10
DEFAULT_SCHEDULER_FACTOR = 0.50
DEFAULT_EARLY_STOP_PATIENCE = 12
DEFAULT_ENCODER_FREEZE_EPOCHS = 2
DEFAULT_ENCODER_LR_SCALE = 0.10


def _group_norm(channels: int) -> nn.GroupNorm:
    # Match the normalization layout stored in masked-model checkpoints
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _PartialConv2d(nn.Module):
    # Recreate one partial convolution from the masked-model checkpoint
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.kernel_area = kernel_size * kernel_size
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("mask_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.convolution(inputs * mask)
        with torch.no_grad():
            support = F.conv2d(mask, self.mask_kernel, padding=self.padding)
            next_mask = (support > 0).to(inputs.dtype)
            scale = self.kernel_area / support.clamp_min(1.0)
        features = (features * scale + self.bias[None, :, None, None]) * next_mask
        return features, next_mask


class _MaskedNO2Stem(nn.Module):
    # Recreate the masked-model NO2 stem for the filling pass
    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.first = _PartialConv2d(1, out_channels, kernel_size=5)
        self.first_norm = _group_norm(out_channels)
        self.second = _PartialConv2d(out_channels, out_channels, kernel_size=3)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features, mask = self.first(values, mask)
        features = self.activation(self.first_norm(features)) * mask
        features, mask = self.second(features, mask)
        return self.activation(self.second_norm(features)) * mask


class _MaskedRasterFrameEncoder(RasterFrameEncoder):
    # Reuse the common encoder layers with the checkpoint's partial-convolution stem
    def __init__(self) -> None:
        super().__init__()
        self.no2_stem = _MaskedNO2Stem(out_channels=16)

    def forward(
        self,
        no2: torch.Tensor,
        weather: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        no2_features = self.no2_stem(no2, visible_mask)
        weather_features = self.weather_stem(weather)
        fused = self.stem_fusion(torch.cat((no2_features, weather_features), dim=1))
        return self.spatial_encoder(fused)


class _MaskedNO2Autoencoder(nn.Module):
    # Rebuild the existing masked model so its checkpoint can fill delta rasters
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _MaskedRasterFrameEncoder()
        self.bottleneck = ResidualBlock(ENCODER_OUTPUT_CHANNELS, ENCODER_OUTPUT_CHANNELS)
        self.decoder_12 = ResidualBlock(ENCODER_OUTPUT_CHANNELS, 48)
        self.decoder_24 = ResidualBlock(48, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        no2 = image[:, :1]
        weather = image[:, 1:MODEL_IMAGE_CHANNELS]
        visible_mask = image[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        latent = self.encoder(no2, weather, visible_mask)
        decoded = self.bottleneck(latent)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = self.decoder_12(decoded)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        return self.output(self.decoder_24(decoded))

    def fill_missing(self, image: torch.Tensor) -> torch.Tensor:
        visible_mask = image[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        return image[:, :1] * visible_mask + self(image) * (1.0 - visible_mask)


def parse_args() -> argparse.Namespace:
    """Parse model-training command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--tabular-epochs", type=int, default=DEFAULT_TABULAR_EPOCHS)
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, NUM_CORES))
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--tabular-learning-rate", type=float, default=DEFAULT_TABULAR_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler-factor", type=float, default=DEFAULT_SCHEDULER_FACTOR)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--pretrained-encoder-weights", default=PRETRAINED_ENCODER_WEIGHTS)
    parser.add_argument("--completed-raster-dir", required=True)
    parser.add_argument("--encoder-freeze-epochs", type=int, default=DEFAULT_ENCODER_FREEZE_EPOCHS)
    parser.add_argument("--encoder-lr-scale", type=float, default=DEFAULT_ENCODER_LR_SCALE)
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


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    # Move model inputs and class labels onto the training device
    image, tabular, elapsed_hours, target, index = batch
    non_blocking = device.type == "cuda"
    return (
        image.to(device, non_blocking=non_blocking),
        tabular.to(device, non_blocking=non_blocking),
        elapsed_hours.to(device, non_blocking=non_blocking),
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
) -> tuple[float, float]:
    """Train one epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        image, tabular, elapsed_hours, target, _ = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(image, tabular, elapsed_hours)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * target.numel()
        correct += int((logits.argmax(dim=1) == target).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch and return mean loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, tabular, elapsed_hours, target, _ = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image, tabular, elapsed_hours)
                loss = criterion(logits, target)
            total_loss += loss.item() * target.numel()
            correct += int((logits.argmax(dim=1) == target).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, object],
    phase_name: str,
    encoder: nn.Module | None = None,
    encoder_freeze_epochs: int = 0,
    encoder_lr_scale: float = 1.0,
) -> tuple[list[float], list[float], float]:
    """Fit one classifier and restore its lowest-validation-loss state."""
    if encoder is None:
        parameter_groups: list[dict[str, object]] = [{"params": model.parameters(), "lr": learning_rate}]
    else:
        encoder_parameters = list(encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        parameter_groups = [
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]},
            {"params": encoder_parameters, "lr": learning_rate * encoder_lr_scale},
        ]
        for parameter in encoder_parameters:
            parameter.requires_grad = encoder_freeze_epochs == 0
    optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if encoder is not None and epoch == encoder_freeze_epochs + 1:
            for parameter in encoder.parameters():
                parameter.requires_grad = True
            print(f"Unfroze pretrained frame encoder at epoch {epoch}")
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.gradient_clip_norm,
        )
        validation_loss, validation_accuracy = val_epoch(model, val_loader, criterion, device)
        scheduler.step(validation_loss)
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        learning_rate_now = optimizer.param_groups[0]["lr"]
        print(
            f"{phase_name} epoch {epoch:03d} | train loss {train_loss:.5f} acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.5f} acc {validation_accuracy:.4f} | lr {learning_rate_now:.2e}"
        )
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping {phase_name} at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return train_losses, val_losses, best_val_loss


def _checkpoint_sha256(path: Path) -> str:
    # Identify the exact masked checkpoint used for reconstruction and transfer
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_masked_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, object], _MaskedNO2Autoencoder]:
    # Load the full reconstruction model and its reusable encoder weights
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["image_keys"]) != tuple(MODEL_IMAGE_KEYS):
        raise ValueError("Masked checkpoint image keys do not match delta raster channels")
    model = _MaskedNO2Autoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def _standard_encoder_state(masked_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Map partial-convolution parameters into the completed-raster convolutional encoder
    standard_state = {}
    for name, value in masked_state.items():
        if name.endswith(".mask_kernel"):
            continue
        standard_name = name.replace(".convolution.weight", ".weight")
        standard_state[standard_name] = value
    return standard_state


def _fill_missing_rasters(
    model: _MaskedNO2Autoencoder,
    datasets: dict[str, NOxDataset],
    *,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Path]:
    # Materialize normalized four-channel rasters before classifier training
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for split, dataset in datasets.items():
            destination_path = output_dir / f"{split}_completed_rasters.npy"
            destination = None
            loader = _loader(dataset, shuffle=False, args=args, device=device)
            for batch_number, batch in enumerate(loader, start=1):
                image, _, _, _, indices = _move_batch(batch, device)
                batch_size, timesteps, channels, height, width = image.shape
                frames = image.reshape(batch_size * timesteps, channels, height, width)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    completed_no2 = model.fill_missing(frames)
                completed_frames = frames[:, :MODEL_IMAGE_CHANNELS].clone()
                completed_frames[:, :1] = completed_no2
                completed = completed_frames.reshape(
                    batch_size,
                    timesteps,
                    MODEL_IMAGE_CHANNELS,
                    height,
                    width,
                )
                completed = completed.float().clamp_(-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS).cpu().numpy()
                if destination is None:
                    destination = np.lib.format.open_memmap(
                        destination_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(dataset), timesteps, MODEL_IMAGE_CHANNELS, height, width),
                    )
                destination[indices.numpy()] = completed
                if batch_number % 100 == 0 or batch_number == len(loader):
                    print(f"Masked fill {split}: {batch_number:,}/{len(loader):,} batches")
            if destination is None:
                raise ValueError(f"Cannot fill empty {split} split")
            destination.flush()
            del destination, loader
            paths[split] = destination_path
    return paths


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate logits and probabilities for one dataset split."""
    model.eval()
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            image, tabular, elapsed_hours, _, index = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image, tabular, elapsed_hours)
            logits = logits.float()
            logits_batches.append(logits.cpu().numpy())
            probability_batches.append(logits.softmax(dim=1).cpu().numpy())
            indices.append(index.numpy())
    return np.concatenate(logits_batches), np.concatenate(probability_batches), np.concatenate(indices)


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
    probabilities: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    # Attach class predictions in source-record order
    frame = dataset.frame.iloc[indices].copy().reset_index(drop=True)
    frame[TRUE_CLASS_COL] = dataset.labels[indices]
    frame[PREDICTED_CLASS_COL] = probabilities.argmax(axis=1)
    for column, values in zip(LOGIT_COLUMNS, logits.T, strict=True):
        frame[column] = values
    for column, values in zip(PROBABILITY_COLUMNS, probabilities.T, strict=True):
        frame[column] = values
    return frame


def _class_counts(dataset: NOxDataset) -> dict[str, int]:
    # Count training examples in the configured class order
    return {name: int(np.sum(dataset.labels == index)) for index, name in enumerate(MODEL_CLASS_NAMES)}


def main() -> None:
    """Train and evaluate raster and tabular classifiers."""
    args = parse_args()
    _seed_everything(args.seed)
    device = _device(args.device)
    completed_root = Path(args.completed_raster_dir).resolve()
    if Path("/tmp") not in completed_root.parents:
        raise ValueError("Completed raster directory must be under job-local /tmp")
    masked_checkpoint_path = Path(args.pretrained_encoder_weights)
    if not args.pretrained_encoder_weights:
        raise ValueError("Set PRETRAINED_ENCODER_WEIGHTS or pass --pretrained-encoder-weights")
    if args.encoder_freeze_epochs < 0 or args.encoder_lr_scale <= 0:
        raise ValueError("Encoder freeze epochs must be nonnegative and its learning-rate scale must be positive")

    run_name = datetime.now(timezone.utc).strftime("delta_category_classification_%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    print(f"Training three-model delta comparison on {device}; outputs: {run_dir}")

    masked_checkpoint, imputation_model = _load_masked_checkpoint(masked_checkpoint_path, device)
    stats = compute_stats("train", fixed_image_stats=masked_checkpoint["normalization_stats"])
    save_stats(stats, run_dir / "normalization_stats.json")

    source_datasets = {split: NOxDataset(split, stats) for split in ("train", "val", "test")}
    masked_checkpoint_sha256 = _checkpoint_sha256(masked_checkpoint_path)
    completed_dir = completed_root / run_name
    completed_paths = _fill_missing_rasters(
        imputation_model,
        source_datasets,
        output_dir=completed_dir,
        args=args,
        device=device,
    )
    del imputation_model, source_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    datasets = {
        split: NOxDataset(split, stats, completed_raster_path=completed_paths[split])
        for split in ("train", "val", "test")
    }
    tabular_datasets = {split: NOxDataset(split, stats, load_images=False) for split in datasets}
    eval_loaders = {
        split: _loader(dataset, shuffle=False, args=args, device=device) for split, dataset in datasets.items()
    }
    tabular_train_loader = _loader(tabular_datasets["train"], shuffle=True, args=args, device=device)
    tabular_eval_loaders = {
        split: _loader(dataset, shuffle=False, args=args, device=device) for split, dataset in tabular_datasets.items()
    }
    common_checkpoint_metadata = {
        "normalization_stats": stats.to_dict(),
        "model_feature_names": MODEL_FEATURE_NAMES,
        "target_name": MODEL_TARGET_COL,
        "class_names": MODEL_CLASS_NAMES,
        "masked_checkpoint_sha256": masked_checkpoint_sha256,
    }
    histories: dict[str, tuple[list[float], list[float]]] = {}
    model_frames: dict[str, dict[str, pd.DataFrame]] = {}
    training_summaries: dict[str, dict[str, object]] = {}

    tabular_model = TabularMLP(len(MODEL_FEATURE_NAMES)).to(device)
    print(f"Training {tabular_model.num_params():,}-parameter tabular classifier on {device}")
    tabular_train_losses, tabular_val_losses, tabular_best_loss = fit_model(
        tabular_model,
        tabular_train_loader,
        tabular_eval_loaders["val"],
        device=device,
        epochs=args.tabular_epochs,
        learning_rate=args.tabular_learning_rate,
        args=args,
        checkpoint_path=checkpoint_dir / "best_tabular_classifier.pt",
        checkpoint_metadata=common_checkpoint_metadata,
        phase_name="Tabular classifier",
    )
    plot_loss_curve(
        tabular_train_losses,
        tabular_val_losses,
        run_dir,
        plot_name="tabular_loss_curve",
        title="Tabular classifier training and validation loss",
    )
    tabular_frames = {}
    for split, loader in tabular_eval_loaders.items():
        logits, probabilities, indices = run_inference(tabular_model, loader, device)
        tabular_frames[split] = _prediction_frame(tabular_datasets[split], logits, probabilities, indices)
    tabular_run = {
        "maximum_epochs": args.tabular_epochs,
        "learning_rate": args.tabular_learning_rate,
        "parameters": tabular_model.num_params(),
        "best_validation_loss": tabular_best_loss,
    }
    histories["mlp"] = (tabular_train_losses, tabular_val_losses)
    model_frames["mlp"] = tabular_frames
    training_summaries["mlp"] = tabular_run
    del tabular_model, tabular_train_loader, tabular_eval_loaders, tabular_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for model_name, use_pretrained_encoder in (
        ("random_init_delta", False),
        ("pretrained_encoder_delta", True),
    ):
        _seed_everything(args.seed)
        raster_model = RasterConvGRUClassifier(head_dim=args.head_dim, dropout=args.dropout).to(device)
        if use_pretrained_encoder:
            raster_model.frame_encoder.load_state_dict(
                _standard_encoder_state(masked_checkpoint["encoder_state_dict"])
            )
        train_loader = _loader(datasets["train"], shuffle=True, args=args, device=device)
        print(f"Training {raster_model.num_params():,}-parameter {model_name} classifier")
        train_losses, val_losses, best_val_loss = fit_model(
            raster_model,
            train_loader,
            eval_loaders["val"],
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            args=args,
            checkpoint_path=checkpoint_dir / f"best_{model_name}.pt",
            checkpoint_metadata={
                **common_checkpoint_metadata,
                "encoder_architecture": ENCODER_ARCHITECTURE_NAME,
                "encoder_initialization": "masked_pretrained" if use_pretrained_encoder else "random",
            },
            phase_name=model_name.replace("_", " ").title(),
            encoder=raster_model.frame_encoder if use_pretrained_encoder else None,
            encoder_freeze_epochs=args.encoder_freeze_epochs if use_pretrained_encoder else 0,
            encoder_lr_scale=args.encoder_lr_scale if use_pretrained_encoder else 1.0,
        )
        histories[model_name] = (train_losses, val_losses)
        raster_frames = {}
        for split, loader in eval_loaders.items():
            logits, probabilities, indices = run_inference(raster_model, loader, device)
            raster_frames[split] = _prediction_frame(datasets[split], logits, probabilities, indices)
        model_frames[model_name] = raster_frames
        training_summaries[model_name] = {
            "maximum_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "parameters": raster_model.num_params(),
            "best_validation_loss": best_val_loss,
            "encoder_initialization": "masked_pretrained" if use_pretrained_encoder else "random",
        }
        plot_loss_curve(
            train_losses,
            val_losses,
            run_dir,
            plot_name=f"{model_name}_loss_curve",
            title=f"{model_name.replace('_', ' ').title()} loss",
        )
        del raster_model, train_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    target_label_mode = str(datasets["train"].frame[LABEL_MODE_COL].iloc[0])
    run_config = {
        "device": str(device),
        "models": ["mlp", "random_init_delta", "pretrained_encoder_delta"],
        "batch_size": args.batch_size,
        "workers": args.workers,
        "maximum_epochs": args.epochs,
        "training": training_summaries,
        "loss_history": {
            name: {"train": train_losses, "validation": validation_losses}
            for name, (train_losses, validation_losses) in histories.items()
        },
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
        "pretrained_encoder_weights": str(masked_checkpoint_path),
        "pretrained_encoder_sha256": masked_checkpoint_sha256,
        "encoder_architecture": ENCODER_ARCHITECTURE_NAME,
        "encoder_freeze_epochs": args.encoder_freeze_epochs,
        "encoder_lr_scale": args.encoder_lr_scale,
        "completed_raster_paths": {split: str(path) for split, path in completed_paths.items()},
        "image_keys": list(stats.image_keys),
        "image_center": list(stats.image_center),
        "image_scale": list(stats.image_scale),
        "image_clip_range": [-MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS],
        "target_name": MODEL_TARGET_COL,
        "class_names": list(MODEL_CLASS_NAMES),
        "class_counts": {split: _class_counts(dataset) for split, dataset in datasets.items()},
        "target_label_mode": target_label_mode,
        "tabular_features": list(MODEL_FEATURE_NAMES),
        "prediction_family": "three_class_classification",
        "sequence_encoder": "completed_raster_convolutional_encoder_then_convgru",
    }
    with (run_dir / "run_config.json").open("w") as destination:
        json.dump(run_config, destination, indent=2)

    plot_training_comparison(
        histories,
        model_frames,
        run_dir,
        encoder_unfreeze_epoch=args.encoder_freeze_epochs + 1,
    )
    plot_confusion_matrices(model_frames, run_dir)
    save_results(model_frames, run_dir, primary_model_name="pretrained_encoder_delta")


if __name__ == "__main__":
    main()
