"""Shared helpers for PINN notebooks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class MLP(nn.Module):
    """Fully connected network: R^d_in -> R^d_out."""

    def __init__(
        self,
        d_in: int = 1,
        d_out: int = 1,
        width: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx for scalar or vector components."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref)
    return float(num / (den + 1e-12))
