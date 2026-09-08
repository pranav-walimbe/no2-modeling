"""Train the XGBoost tabular baseline for NOx-change classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import LABEL_COL
from modeling.dataset import MODEL_FEATURE_NAMES, NOxDataset
from modeling.eval_utils import (
    LOGIT_COL,
    POSITIVE_PROBABILITY_COL,
    PREDICTED_CLASS_COL,
    TRUE_CLASS_COL,
)

DEFAULT_ESTIMATORS = 1_000
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_CHILD_WEIGHT = 5.0
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_COLUMN_SUBSAMPLE = 0.8
DEFAULT_REGULARIZATION = 1.0
DEFAULT_EARLY_STOPPING_ROUNDS = 25
PROBABILITY_EPSILON = 1e-7


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for the tabular baseline."""

    n_estimators: int = DEFAULT_ESTIMATORS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_depth: int = DEFAULT_MAX_DEPTH
    min_child_weight: float = DEFAULT_MIN_CHILD_WEIGHT
    subsample: float = DEFAULT_SUBSAMPLE
    colsample_bytree: float = DEFAULT_COLUMN_SUBSAMPLE
    reg_lambda: float = DEFAULT_REGULARIZATION
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-safe baseline settings.

        Returns:
            XGBoost training settings.
        """
        return asdict(self)


@dataclass(frozen=True)
class XGBoostRun:
    """Trained baseline outputs for one modeling run."""

    split_frames: dict[str, pd.DataFrame]
    config: XGBoostConfig
    best_iteration: int
    best_validation_logloss: float


def _feature_frame(dataset: NOxDataset) -> pd.DataFrame:
    # Retain feature names in the fitted booster and its saved artifact
    return pd.DataFrame(dataset.features, columns=MODEL_FEATURE_NAMES)


def _prediction_frame(dataset: NOxDataset, classifier: XGBClassifier) -> pd.DataFrame:
    # Attach baseline probabilities and classes in source-record order
    probability = classifier.predict_proba(_feature_frame(dataset))[:, 1].astype(np.float64)
    clipped_probability = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    frame = dataset.frame.copy().reset_index(drop=True)
    frame[TRUE_CLASS_COL] = frame[LABEL_COL].to_numpy(dtype=np.uint8)
    frame[LOGIT_COL] = np.log(clipped_probability / (1.0 - clipped_probability))
    frame[POSITIVE_PROBABILITY_COL] = probability
    frame[PREDICTED_CLASS_COL] = (probability >= 0.5).astype(np.uint8)
    return frame


def train_xgboost_baseline(
    datasets: dict[str, NOxDataset],
    checkpoint_path: str | Path,
    *,
    seed: int,
    workers: int,
    config: XGBoostConfig | None = None,
) -> XGBoostRun:
    """Train and evaluate XGBoost on tabular inputs from frozen splits.

    Args:
        datasets: Train, validation, and test datasets with shared normalization.
        checkpoint_path: Destination for the fitted booster.
        seed: Random seed for row and feature sampling.
        workers: Maximum CPU threads used by XGBoost.
        config: Optional baseline hyperparameters.

    Returns:
        Baseline predictions and validation-selected training details.
    """
    resolved_config = config or XGBoostConfig()
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=max(workers, 1),
        **resolved_config.to_dict(),
    )
    train_dataset = datasets["train"]
    validation_dataset = datasets["val"]
    classifier.fit(
        _feature_frame(train_dataset),
        train_dataset.labels,
        eval_set=[(_feature_frame(validation_dataset), validation_dataset.labels)],
        verbose=False,
    )
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(destination)
    split_frames = {name: _prediction_frame(dataset, classifier) for name, dataset in datasets.items()}
    return XGBoostRun(
        split_frames=split_frames,
        config=resolved_config,
        best_iteration=int(classifier.best_iteration),
        best_validation_logloss=float(classifier.best_score),
    )
