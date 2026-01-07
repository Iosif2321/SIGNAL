"""Feature importance helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def _get_first_layer_weight(model: torch.nn.Module) -> np.ndarray:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return module.weight.detach().cpu().numpy()
    raise ValueError("No Linear layer found in model.")


def compute_feature_importance(
    model: torch.nn.Module,
    feature_cols: List[str],
    window_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean abs weight importance per feature/lag and aggregated by feature."""
    weight = _get_first_layer_weight(model)
    importance = np.mean(np.abs(weight), axis=0)

    cols = []
    for lag in range(window_size):
        lag_idx = window_size - 1 - lag
        for feature in feature_cols:
            cols.append((feature, lag_idx))

    if len(cols) != len(importance):
        raise ValueError("Feature mapping size mismatch.")

    df = pd.DataFrame(
        {
            "feature": [c[0] for c in cols],
            "lag": [c[1] for c in cols],
            "importance": importance.astype(float),
        }
    )
    df_agg = df.groupby("feature", as_index=False)["importance"].mean()
    return df, df_agg