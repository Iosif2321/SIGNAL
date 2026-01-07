"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    start_ms: Optional[int]
    end_ms: Optional[int]
    start_date: Optional[str]
    end_date: Optional[str]
    limit_per_call: int
    output_path: str


@dataclass(frozen=True)
class ParityConfig:
    duration_sec: int
    ws_topic: str
    rest_compare_mode: str


@dataclass(frozen=True)
class FeaturesConfig:
    window_size_K: int
    list_of_features: List[str]


@dataclass(frozen=True)
class SupervisedConfig:
    epochs: int
    batch_size: int
    lr: float
    early_stopping_patience: int


@dataclass(frozen=True)
class RewardConfig:
    R_correct: float
    R_wrong: float
    R_hold: float


@dataclass(frozen=True)
class RLConfig:
    episodes: int
    steps_per_episode: int
    gamma: float
    lr: float
    reward: RewardConfig
    entropy_bonus: float


@dataclass(frozen=True)
class DecisionRuleConfig:
    T_min: float


@dataclass(frozen=True)
class VizConfig:
    out_dir: str
    moving_window: int
    save_formats: List[str]


@dataclass(frozen=True)
class Config:
    symbol: str
    category: str
    interval: str
    dataset: DatasetConfig
    parity: ParityConfig
    features: FeaturesConfig
    supervised: SupervisedConfig
    rl: RLConfig
    decision_rule: DecisionRuleConfig
    viz: VizConfig


def _parse_date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _maybe_date_to_ms(date_str: Optional[str]) -> Optional[int]:
    if date_str is None:
        return None
    return _parse_date_to_ms(date_str)


def load_config(path: str | Path) -> Config:
    """Load YAML config and return validated Config."""
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text())

    dataset = data.get("dataset", {})
    start_ms = dataset.get("start_ms")
    end_ms = dataset.get("end_ms")
    start_date = dataset.get("start_date")
    end_date = dataset.get("end_date")

    if start_ms is None and start_date is not None:
        start_ms = _maybe_date_to_ms(start_date)
    if end_ms is None and end_date is not None:
        end_ms = _maybe_date_to_ms(end_date)

    output_path = str(dataset["output_path"])
    viz_out_dir = str(data["viz"]["out_dir"])
    run_root = os.environ.get("CRYPTOMVP_RUN_DIR")
    if run_root:
        if not Path(output_path).is_absolute():
            output_path = str(Path(run_root) / output_path)
        if not Path(viz_out_dir).is_absolute():
            viz_out_dir = str(Path(run_root) / viz_out_dir)

    dataset_cfg = DatasetConfig(
        start_ms=start_ms,
        end_ms=end_ms,
        start_date=start_date,
        end_date=end_date,
        limit_per_call=int(dataset["limit_per_call"]),
        output_path=output_path,
    )

    parity_cfg = ParityConfig(
        duration_sec=int(data["parity"]["duration_sec"]),
        ws_topic=str(data["parity"]["ws_topic"]),
        rest_compare_mode=str(data["parity"]["rest_compare_mode"]),
    )

    features_cfg = FeaturesConfig(
        window_size_K=int(data["features"]["window_size_K"]),
        list_of_features=list(data["features"]["list_of_features"]),
    )

    supervised_cfg = SupervisedConfig(
        epochs=int(data["supervised"]["epochs"]),
        batch_size=int(data["supervised"]["batch_size"]),
        lr=float(data["supervised"]["lr"]),
        early_stopping_patience=int(data["supervised"]["early_stopping_patience"]),
    )

    reward_cfg = RewardConfig(
        R_correct=float(data["rl"]["reward"]["R_correct"]),
        R_wrong=float(data["rl"]["reward"]["R_wrong"]),
        R_hold=float(data["rl"]["reward"]["R_hold"]),
    )

    rl_cfg = RLConfig(
        episodes=int(data["rl"]["episodes"]),
        steps_per_episode=int(data["rl"]["steps_per_episode"]),
        gamma=float(data["rl"]["gamma"]),
        lr=float(data["rl"]["lr"]),
        reward=reward_cfg,
        entropy_bonus=float(data["rl"]["entropy_bonus"]),
    )

    decision_cfg = DecisionRuleConfig(T_min=float(data["decision_rule"]["T_min"]))

    viz_cfg = VizConfig(
        out_dir=viz_out_dir,
        moving_window=int(data["viz"]["moving_window"]),
        save_formats=list(data["viz"]["save_formats"]),
    )

    return Config(
        symbol=str(data["symbol"]),
        category=str(data["category"]),
        interval=str(data["interval"]),
        dataset=dataset_cfg,
        parity=parity_cfg,
        features=features_cfg,
        supervised=supervised_cfg,
        rl=rl_cfg,
        decision_rule=decision_cfg,
        viz=viz_cfg,
    )


def override_config(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Create a new Config with simple top-level overrides for fast mode."""
    data = cfg.__dict__.copy()
    for key, value in overrides.items():
        if key in data:
            data[key] = value
    return Config(**data)
