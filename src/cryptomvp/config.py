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
    target_closed_candles: int
    max_wait_sec: int


@dataclass(frozen=True)
class FeaturesConfig:
    window_size_K: int
    list_of_features: List[str]
    feature_set_id: Optional[str] = None
    feature_sets_path: Optional[str] = None


@dataclass(frozen=True)
class SupervisedConfig:
    epochs: int
    batch_size: int
    lr: float
    early_stopping_patience: int
    weight_decay: float
    hidden_dim: int


@dataclass(frozen=True)
class RewardConfig:
    R_correct: float
    R_wrong: float
    R_opposite: float
    margin_threshold: float
    margin_penalty: float


@dataclass(frozen=True)
class RLConfig:
    episodes: int
    steps_per_episode: int
    gamma: float
    lr: float
    reward: RewardConfig
    entropy_bonus: float
    policy_hidden_dim: int


@dataclass(frozen=True)
class DecisionRuleConfig:
    T_min: float
    delta_min: float
    delta_grid: List[float]
    use_best_from_scan: bool
    scan_min: float
    scan_max: float
    scan_step: float


@dataclass(frozen=True)
class VizConfig:
    out_dir: str
    moving_window: int
    save_formats: List[str]


@dataclass(frozen=True)
class TunerRewardConfig:
    decision_accuracy_weight: float
    decision_action_rate_weight: float
    decision_conflict_penalty: float
    decision_hold_penalty: float
    rl_up_accuracy_weight: float
    rl_down_accuracy_weight: float
    rl_up_error_balance_penalty: float
    rl_down_error_balance_penalty: float
    rl_up_hold_penalty: float
    rl_down_hold_penalty: float
    improve_up_accuracy_weight: float
    improve_down_accuracy_weight: float
    decision_min_session_accuracy_weight: float = 0.0
    decision_min_session_precision_up_weight: float = 0.0
    decision_min_session_precision_down_weight: float = 0.0
    decision_max_session_hold_penalty: float = 0.0
    decision_max_session_conflict_penalty: float = 0.0


@dataclass(frozen=True)
class TunerCandidateConfig:
    name: str
    features: List[str]
    T_min: float
    delta_min: float
    lr: float
    weight_decay: float
    hidden_dim: int


@dataclass(frozen=True)
class TunerSearchConfig:
    explore_prob: float
    mutate_prob: float
    max_mutations: int
    neighbor_only: bool
    always_mutate: bool


@dataclass(frozen=True)
class TunerFeatureSelectionConfig:
    enabled: bool
    top_n: int
    corr_threshold: float
    var_threshold: float


@dataclass(frozen=True)
class TunerConfig:
    episodes: int
    entropy_bonus: float
    lr: float
    reward: TunerRewardConfig
    param_space: Dict[str, List[Any]]
    candidates: Optional[List[TunerCandidateConfig]]
    search: TunerSearchConfig
    feature_selection: Optional[TunerFeatureSelectionConfig] = None


@dataclass(frozen=True)
class AdaptationConfig:
    min_action_accuracy: Optional[float]
    min_action_rate: Optional[float]
    max_hold_rate: Optional[float]
    max_conflict_rate: Optional[float]
    min_precision_up: Optional[float]
    min_precision_down: Optional[float]
    min_rl_up_accuracy: Optional[float]
    min_rl_down_accuracy: Optional[float]
    max_rl_up_hold_rate: Optional[float]
    max_rl_down_hold_rate: Optional[float]
    max_drift_score: Optional[float] = None


@dataclass(frozen=True)
class SessionOverrideConfig:
    tz: Optional[str]
    start: Optional[str]
    end: Optional[str]
    overrides: Dict[str, Any]


@dataclass(frozen=True)
class SessionConfig:
    mode: str
    overlap_policy: str
    priority_order: List[str]
    strategy: str
    sessions: Dict[str, SessionOverrideConfig]


@dataclass(frozen=True)
class Config:
    symbol: str
    category: str
    interval: str
    seed: int
    device: str
    allow_cpu_fallback: bool
    dataset: DatasetConfig
    parity: ParityConfig
    features: FeaturesConfig
    supervised: SupervisedConfig
    rl: RLConfig
    decision_rule: DecisionRuleConfig
    viz: VizConfig
    tuner: Optional[TunerConfig]
    adaptation: Optional[AdaptationConfig]
    session: Optional[SessionConfig]


def _parse_date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _maybe_date_to_ms(date_str: Optional[str]) -> Optional[int]:
    if date_str is None:
        return None
    return _parse_date_to_ms(date_str)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def load_config(path: str | Path) -> Config:
    """Load YAML config and return validated Config."""
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text())
    device = str(data.get("device", "auto"))
    allow_cpu_fallback = bool(data.get("allow_cpu_fallback", False))

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
        target_closed_candles=int(data["parity"].get("target_closed_candles", 10)),
        max_wait_sec=int(data["parity"].get("max_wait_sec", data["parity"]["duration_sec"])),
    )

    features_cfg = FeaturesConfig(
        window_size_K=int(data["features"]["window_size_K"]),
        list_of_features=list(data["features"]["list_of_features"]),
        feature_set_id=data["features"].get("feature_set_id"),
        feature_sets_path=data["features"].get("feature_sets_path"),
    )

    supervised_cfg = SupervisedConfig(
        epochs=int(data["supervised"]["epochs"]),
        batch_size=int(data["supervised"]["batch_size"]),
        lr=float(data["supervised"]["lr"]),
        early_stopping_patience=int(data["supervised"]["early_stopping_patience"]),
        weight_decay=float(data["supervised"].get("weight_decay", 0.0)),
        hidden_dim=int(data["supervised"].get("hidden_dim", 64)),
    )

    reward_cfg = RewardConfig(
        R_correct=float(data["rl"]["reward"]["R_correct"]),
        R_wrong=float(data["rl"]["reward"]["R_wrong"]),
        R_opposite=float(data["rl"]["reward"].get("R_opposite", data["rl"]["reward"]["R_wrong"])),
        margin_threshold=float(data["rl"]["reward"].get("margin_threshold", 0.0)),
        margin_penalty=float(data["rl"]["reward"].get("margin_penalty", 0.0)),
    )
    if (
        reward_cfg.R_correct <= 0
        or reward_cfg.R_wrong <= 0
        or reward_cfg.R_opposite <= 0
        or reward_cfg.margin_threshold < 0
        or reward_cfg.margin_penalty < 0
    ):
        raise ValueError(
            "Reward config values must be positive (R_correct/R_wrong/R_opposite) "
            "and non-negative (margin_threshold/margin_penalty)."
        )

    rl_cfg = RLConfig(
        episodes=int(data["rl"]["episodes"]),
        steps_per_episode=int(data["rl"]["steps_per_episode"]),
        gamma=float(data["rl"]["gamma"]),
        lr=float(data["rl"]["lr"]),
        reward=reward_cfg,
        entropy_bonus=float(data["rl"]["entropy_bonus"]),
        policy_hidden_dim=int(data["rl"].get("policy_hidden_dim", 64)),
    )
    if rl_cfg.entropy_bonus < 0:
        raise ValueError("entropy_bonus must be non-negative.")

    decision_cfg = DecisionRuleConfig(
        T_min=float(data["decision_rule"]["T_min"]),
        delta_min=float(data["decision_rule"].get("delta_min", 0.0)),
        delta_grid=[
            float(val)
            for val in data["decision_rule"].get("delta_grid", [data["decision_rule"].get("delta_min", 0.0)])
        ],
        use_best_from_scan=bool(data["decision_rule"].get("use_best_from_scan", True)),
        scan_min=float(data["decision_rule"].get("scan_min", 0.45)),
        scan_max=float(data["decision_rule"].get("scan_max", 0.75)),
        scan_step=float(data["decision_rule"].get("scan_step", 0.01)),
    )

    viz_cfg = VizConfig(
        out_dir=viz_out_dir,
        moving_window=int(data["viz"]["moving_window"]),
        save_formats=list(data["viz"]["save_formats"]),
    )

    tuner_cfg = None
    if "tuner" in data:
        tuner = data["tuner"]
        reward_cfg = TunerRewardConfig(
            decision_accuracy_weight=float(tuner["reward"]["decision_accuracy_weight"]),
            decision_action_rate_weight=float(tuner["reward"]["decision_action_rate_weight"]),
            decision_conflict_penalty=float(tuner["reward"]["decision_conflict_penalty"]),
            decision_hold_penalty=float(tuner["reward"]["decision_hold_penalty"]),
            rl_up_accuracy_weight=float(tuner["reward"].get("rl_up_accuracy_weight", 0.0)),
            rl_down_accuracy_weight=float(tuner["reward"].get("rl_down_accuracy_weight", 0.0)),
            rl_up_error_balance_penalty=float(
                tuner["reward"].get("rl_up_error_balance_penalty", 0.0)
            ),
            rl_down_error_balance_penalty=float(
                tuner["reward"].get("rl_down_error_balance_penalty", 0.0)
            ),
            rl_up_hold_penalty=float(tuner["reward"].get("rl_up_hold_penalty", 0.0)),
            rl_down_hold_penalty=float(tuner["reward"].get("rl_down_hold_penalty", 0.0)),
            improve_up_accuracy_weight=float(
                tuner["reward"].get("improve_up_accuracy_weight", 0.0)
            ),
            improve_down_accuracy_weight=float(
                tuner["reward"].get("improve_down_accuracy_weight", 0.0)
            ),
            decision_min_session_accuracy_weight=float(
                tuner["reward"].get("decision_min_session_accuracy_weight", 0.0)
            ),
            decision_min_session_precision_up_weight=float(
                tuner["reward"].get("decision_min_session_precision_up_weight", 0.0)
            ),
            decision_min_session_precision_down_weight=float(
                tuner["reward"].get("decision_min_session_precision_down_weight", 0.0)
            ),
            decision_max_session_hold_penalty=float(
                tuner["reward"].get("decision_max_session_hold_penalty", 0.0)
            ),
            decision_max_session_conflict_penalty=float(
                tuner["reward"].get("decision_max_session_conflict_penalty", 0.0)
            ),
        )
        candidates = None
        if "candidates" in tuner:
            cand_list: List[TunerCandidateConfig] = []
            for cand in tuner["candidates"]:
                cand_list.append(
                    TunerCandidateConfig(
                        name=str(cand.get("name", "candidate")),
                        features=list(cand["features"]),
                        T_min=float(cand["T_min"]),
                        delta_min=float(cand.get("delta_min", 0.0)),
                        lr=float(cand.get("lr", data["supervised"]["lr"])),
                        weight_decay=float(cand.get("weight_decay", data["supervised"].get("weight_decay", 0.0))),
                        hidden_dim=int(cand.get("hidden_dim", 64)),
                    )
                )
            candidates = cand_list
        param_space: Dict[str, List[Any]] = {}
        for key, values in tuner.get("param_space", {}).items():
            if isinstance(values, list):
                param_space[str(key)] = list(values)
            else:
                param_space[str(key)] = [values]
        search = tuner.get("search", {})
        search_cfg = TunerSearchConfig(
            explore_prob=float(search.get("explore_prob", 0.1)),
            mutate_prob=float(search.get("mutate_prob", 0.5)),
            max_mutations=int(search.get("max_mutations", 2)),
            neighbor_only=bool(search.get("neighbor_only", False)),
            always_mutate=bool(search.get("always_mutate", True)),
        )
        feature_selection_cfg = None
        if "feature_selection" in tuner:
            feat = tuner.get("feature_selection", {}) or {}
            feature_selection_cfg = TunerFeatureSelectionConfig(
                enabled=bool(feat.get("enabled", False)),
                top_n=int(feat.get("top_n", 10)),
                corr_threshold=float(feat.get("corr_threshold", 0.98)),
                var_threshold=float(feat.get("var_threshold", 1e-12)),
            )
        tuner_cfg = TunerConfig(
            episodes=int(tuner["episodes"]),
            entropy_bonus=float(tuner.get("entropy_bonus", 0.0)),
            lr=float(tuner.get("lr", 0.1)),
            reward=reward_cfg,
            param_space=param_space,
            candidates=candidates,
            search=search_cfg,
            feature_selection=feature_selection_cfg,
        )

    adaptation_cfg = None
    if "adaptation" in data:
        adapt = data["adaptation"]
        adaptation_cfg = AdaptationConfig(
            min_action_accuracy=_maybe_float(adapt.get("min_action_accuracy")),
            min_action_rate=_maybe_float(adapt.get("min_action_rate")),
            max_hold_rate=_maybe_float(adapt.get("max_hold_rate")),
            max_conflict_rate=_maybe_float(adapt.get("max_conflict_rate")),
            min_precision_up=_maybe_float(adapt.get("min_precision_up")),
            min_precision_down=_maybe_float(adapt.get("min_precision_down")),
            min_rl_up_accuracy=_maybe_float(adapt.get("min_rl_up_accuracy")),
            min_rl_down_accuracy=_maybe_float(adapt.get("min_rl_down_accuracy")),
            max_rl_up_hold_rate=_maybe_float(adapt.get("max_rl_up_hold_rate")),
            max_rl_down_hold_rate=_maybe_float(adapt.get("max_rl_down_hold_rate")),
            max_drift_score=_maybe_float(adapt.get("max_drift_score")),
        )

    session_cfg = None
    if "session" in data:
        sess = data["session"] or {}
        sessions: Dict[str, SessionOverrideConfig] = {}
        for name, sdata in (sess.get("sessions", {}) or {}).items():
            sessions[str(name)] = SessionOverrideConfig(
                tz=sdata.get("tz"),
                start=sdata.get("start"),
                end=sdata.get("end"),
                overrides=dict(sdata.get("overrides", {}) or {}),
            )
        session_cfg = SessionConfig(
            mode=str(sess.get("mode", "fixed_utc_partitions")),
            overlap_policy=str(sess.get("overlap_policy", "priority")),
            priority_order=list(sess.get("priority_order", ["US", "EUROPE", "ASIA"])),
            strategy=str(sess.get("strategy", "experts_per_session")),
            sessions=sessions,
        )

    return Config(
        symbol=str(data["symbol"]),
        category=str(data["category"]),
        interval=str(data["interval"]),
        seed=int(data.get("seed", 42)),
        device=device,
        allow_cpu_fallback=allow_cpu_fallback,
        dataset=dataset_cfg,
        parity=parity_cfg,
        features=features_cfg,
        supervised=supervised_cfg,
        rl=rl_cfg,
        decision_rule=decision_cfg,
        viz=viz_cfg,
        tuner=tuner_cfg,
        adaptation=adaptation_cfg,
        session=session_cfg,
    )


def override_config(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Create a new Config with simple top-level overrides for fast mode."""
    data = cfg.__dict__.copy()
    for key, value in overrides.items():
        if key in data:
            data[key] = value
    return Config(**data)
