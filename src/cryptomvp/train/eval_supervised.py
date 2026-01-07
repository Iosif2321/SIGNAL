"""Supervised evaluation utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from cryptomvp.utils.gpu import require_cuda


def evaluate_supervised(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
) -> Dict[str, object]:
    device = require_cuda()
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size)

    model = model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    acc = float(accuracy_score(targets_np, preds_np))
    f1 = float(f1_score(targets_np, preds_np, zero_division=0))
    cm = confusion_matrix(targets_np, preds_np)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "probs": probs_np,
        "preds": preds_np,
        "targets": targets_np,
    }