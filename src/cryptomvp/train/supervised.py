"""Supervised training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cryptomvp.train.diagnostics import params_vector
from cryptomvp.utils.gpu import require_cuda
from cryptomvp.utils.io import checkpoints_dir
from cryptomvp.utils.logging import get_logger


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    weight_norms: List[float]
    delta_weight_norms: List[float]
    layer_weight_norms: List[Dict[str, float]]


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def train_supervised(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    weight_decay: float,
    model_name: str,
    track_weights: bool = False,
) -> TrainHistory:
    """Train a supervised model on GPU-only."""
    device = require_cuda()
    logger = get_logger(f"train.{model_name}")

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = TrainHistory([], [], [], [], [], [], [])
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    prev_params = params_vector(model).detach().clone() if track_weights else None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accs = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_accs.append(_accuracy(logits, yb))

        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))
                val_accs.append(_accuracy(logits, yb))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        if track_weights:
            curr_params = params_vector(model).detach()
            weight_norm = float(torch.norm(curr_params, p=2).item())
            delta_norm = float(torch.norm(curr_params - prev_params, p=2).item()) if prev_params is not None else 0.0
            layer_norms: Dict[str, float] = {}
            for name, param in model.named_parameters():
                if param.ndim >= 2:
                    layer_norms[name] = float(torch.norm(param.detach(), p=2).item())
            history.weight_norms.append(weight_norm)
            history.delta_weight_norms.append(delta_norm)
            history.layer_weight_norms.append(layer_norms)
            prev_params = curr_params.clone()

        logger.info(
            "Epoch %s/%s - train_loss=%.4f val_loss=%.4f train_acc=%.3f val_acc=%.3f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            ckpt_path = checkpoints_dir() / f"{model_name}.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
