"""Compact tabular model for emissions-change prediction."""

import torch
from torch import nn

TABULAR_HIDDEN_DIM = 32
TABULAR_EMBEDDING_DIM = 16


class TabularMLP(nn.Module):
    """Independently trainable tabular classifier with a reusable embedding."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = TABULAR_HIDDEN_DIM,
        embedding_dim: int = TABULAR_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, tabular: torch.Tensor) -> torch.Tensor:
        """Produce the embedding used by the classifier and fused model."""
        return self.encoder(tabular)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        elapsed_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one Bernoulli logit from tabular features."""
        del image, elapsed_hours
        return self.classifier(self.encode(tabular)).squeeze(1)

    def num_params(self) -> int:
        """Count trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
