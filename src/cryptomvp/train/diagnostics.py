"""Diagnostics for gradients and weights."""

from __future__ import annotations

from typing import Optional

import torch


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(torch.norm(param.grad.detach(), p=2).item() ** 2)
    return float(total ** 0.5)


def params_vector(model: torch.nn.Module) -> torch.Tensor:
    params = [p.detach().flatten() for p in model.parameters()]
    if not params:
        return torch.tensor([])
    return torch.cat(params)


def weight_delta_norm(prev_params: Optional[torch.Tensor], model: torch.nn.Module) -> float:
    curr = params_vector(model)
    if prev_params is None or prev_params.numel() == 0:
        return 0.0
    return float(torch.norm(curr - prev_params, p=2).item())