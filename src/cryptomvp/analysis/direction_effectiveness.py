"""Direction effectiveness analysis for UP/DOWN models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt

from cryptomvp.config import Config, load_config
from cryptomvp.data.features import compute_features
from cryptomvp.data.labels import make_up_down_labels
from cryptomvp.data.windowing import make_windows
from cryptomvp.utils.logging import get_logger
from cryptomvp.viz.plotting import (
    plot_confusion_matrix,
    plot_histogram,
    plot_series_with_band,
    save_figure,
)
from cryptomvp.viz.style import apply_style


Y_TRUE_CANDIDATES = [
    "y_true",
    "true",
    "target",
    "label",
    "direction_true",
    "true_direction",
]
Y_PRED_CANDIDATES = ["y_pred", "pred", "prediction", "decision", "signal", "action"]
Y_PROB_CANDIDATES = [
    "prob",
    "proba",
    "y_prob",
    "pred_prob",
    "score",
    "p",
    "p_up",
    "p_down",
    "p_direction",
]
TIME_CANDIDATES = ["open_time_ms", "time_ms", "timestamp", "time", "open_time"]


@dataclass(frozen=True)
class AnalysisInputs:
    run_dir: Path
    out_dir: Path
    threshold: float
    rolling_window: int
    formats: Sequence[str]
    scan_thresholds: bool
    t_min: float
    t_max: float
    t_step: float
    config_path: Path


def list_run_artifacts(run_dir: Path) -> Dict[str, Path]:
    """Return known artifact paths for a run."""
    return {
        "supervised_up": run_dir / "reports/supervised_up/decision_log.parquet",
        "supervised_down": run_dir / "reports/supervised_down/decision_log.parquet",
        "rl_up_episode": run_dir / "reports/rl_up/episode_metrics.csv",
        "rl_up_step": run_dir / "reports/rl_up/step_log.parquet",
        "rl_down_episode": run_dir / "reports/rl_down/episode_metrics.csv",
        "rl_down_step": run_dir / "reports/rl_down/step_log.parquet",
        "decision_rule": run_dir / "reports/decision_rule/decision_log.parquet",
    }


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a parquet/csv file."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def infer_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    """Infer a column name from candidates with case-insensitive matching."""
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lowered = cand.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    if required:
        raise ValueError(
            f"Missing column for {label}. Candidates={candidates}. Columns={list(df.columns)}"
        )
    return None


def resolve_dataset_path(cfg: Config, run_dir: Path) -> Optional[Path]:
    """Resolve dataset path from config and run dir."""
    cfg_path = Path(cfg.dataset.output_path)
    if cfg_path.is_absolute() and cfg_path.exists():
        return cfg_path
    candidate = run_dir / cfg.dataset.output_path
    if candidate.exists():
        return candidate
    if cfg_path.exists():
        return cfg_path
    return None


def load_rl_labels(cfg: Config, run_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute labels from the run dataset for RL recall metrics."""
    dataset_path = resolve_dataset_path(cfg, run_dir)
    if dataset_path is None:
        return None, None
    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    features = compute_features(df, cfg.features.list_of_features)
    _, window_times, _ = make_windows(features, cfg.features.window_size_K)
    y_up, y_down = make_up_down_labels(df, window_times)
    return y_up, y_down


def _time_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    time_col = infer_column(df, TIME_CANDIDATES, "time", required=False)
    if time_col is None:
        return np.arange(len(df)), "Index"
    return df[time_col].to_numpy(dtype=float), time_col


def _plot_curve(
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    out_base: Path,
    formats: Iterable[str],
    baseline: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    baseline_label: str = "baseline",
) -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    if baseline is not None:
        base_x, base_y = baseline
        ax.plot(base_x, base_y, linestyle="--", label=baseline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_figure(fig, out_base, formats)
    plt.close(fig)


def _normalize_direction(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        mapping = {1: "UP", -1: "DOWN", 0: "HOLD"}
        return series.map(mapping).fillna("HOLD").astype(str)
    normalized = series.astype(str).str.upper().str.strip()
    return normalized.replace({"FLAT": "HOLD"})


def _safe_metrics_from_probs(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    if len(np.unique(y_true)) < 2:
        return None, None, None, None
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return roc, pr, (fpr, tpr), (prec, rec)


def analyze_supervised(
    name: str,
    df: pd.DataFrame,
    threshold: float,
    out_dir: Path,
    formats: Sequence[str],
    rolling_window: int,
) -> Dict[str, float]:
    """Analyze supervised decision logs and generate plots."""
    prob_col = infer_column(df, Y_PROB_CANDIDATES, "probability")
    true_col = infer_column(df, Y_TRUE_CANDIDATES, "y_true")
    y_true = df[true_col].astype(int).to_numpy()
    y_prob = df[prob_col].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    coverage = float(np.mean(y_pred == 1))

    roc_auc, pr_auc, roc_curve_data, pr_curve_data = _safe_metrics_from_probs(
        y_true, y_prob
    )

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    time_x, time_label = _time_axis(df)

    plot_series_with_band(
        time_x,
        y_prob,
        window=rolling_window,
        title=f"{name} Probability Over Time",
        xlabel=time_label,
        ylabel="Probability",
        label="prob",
        out_base=fig_dir / f"{name.lower()}_prob_time",
        formats=formats,
    )

    accuracy_series = (y_pred == y_true).astype(float)
    plot_series_with_band(
        time_x,
        accuracy_series,
        window=rolling_window,
        title=f"{name} Accuracy Over Time",
        xlabel=time_label,
        ylabel="Accuracy",
        label="accuracy",
        out_base=fig_dir / f"{name.lower()}_accuracy_time",
        formats=formats,
    )

    plot_histogram(
        y_prob,
        bins=30,
        title=f"{name} Probability Histogram",
        xlabel="Probability",
        ylabel="Count",
        out_base=fig_dir / f"{name.lower()}_prob_hist",
        formats=formats,
    )

    plot_confusion_matrix(
        cm,
        labels=["0", "1"],
        title=f"{name} Confusion Matrix",
        out_base=fig_dir / f"{name.lower()}_confusion_matrix",
        formats=formats,
    )

    if roc_curve_data is not None:
        fpr, tpr = roc_curve_data
        _plot_curve(
            fpr,
            tpr,
            title=f"{name} ROC Curve",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            label=f"AUC={roc_auc:.3f}" if roc_auc is not None else "ROC",
            out_base=fig_dir / f"{name.lower()}_roc_curve",
            formats=formats,
            baseline=([0, 1], [0, 1]),
            baseline_label="random",
        )
    if pr_curve_data is not None:
        prec, rec = pr_curve_data
        _plot_curve(
            rec,
            prec,
            title=f"{name} PR Curve",
            xlabel="Recall",
            ylabel="Precision",
            label=f"AP={pr_auc:.3f}" if pr_auc is not None else "PR",
            out_base=fig_dir / f"{name.lower()}_pr_curve",
            formats=formats,
        )

    prob_mean = float(np.mean(y_prob))
    prob_std = float(np.std(y_prob))

    return {
        "model": name,
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else np.nan,
        "pr_auc": float(pr_auc) if pr_auc is not None else np.nan,
        "coverage": coverage,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prob_mean": prob_mean,
        "prob_std": prob_std,
    }


def _decision_rule_from_probs(
    p_up: np.ndarray, p_down: np.ndarray, threshold: float
) -> np.ndarray:
    decision = np.full(len(p_up), "HOLD", dtype=object)
    max_prob = np.maximum(p_up, p_down)
    take = max_prob >= threshold
    direction = np.where(p_up >= p_down, "UP", "DOWN")
    decision[take] = direction[take]
    return decision


def analyze_decision_rule(
    df: pd.DataFrame,
    threshold: float,
    out_dir: Path,
    formats: Sequence[str],
    rolling_window: int,
    scan_thresholds: bool,
    t_min: float,
    t_max: float,
    t_step: float,
) -> Dict[str, float]:
    """Analyze combined decision rule logs."""
    p_up_col = infer_column(df, ["p_up", "prob_up", "up_prob"], "p_up")
    p_down_col = infer_column(df, ["p_down", "prob_down", "down_prob"], "p_down")
    decision_col = infer_column(df, ["decision", "signal", "action"], "decision", required=False)
    true_col = infer_column(df, ["true_direction", "direction_true", "true"], "true_direction", required=False)
    correct_col = infer_column(df, ["correct_direction", "correct"], "correct_direction", required=False)

    p_up = df[p_up_col].astype(float).to_numpy()
    p_down = df[p_down_col].astype(float).to_numpy()

    if decision_col is not None:
        decision = _normalize_direction(df[decision_col])
    else:
        decision = pd.Series(_decision_rule_from_probs(p_up, p_down, threshold))

    is_hold = decision.eq("HOLD")
    hold_rate = float(is_hold.mean())
    action_rate = float((~is_hold).mean())
    conflict_rate = float(np.mean((p_up >= threshold) & (p_down >= threshold)))

    action_accuracy = np.nan
    if true_col is not None:
        true_dir = _normalize_direction(df[true_col])
        mask = ~is_hold
        if mask.any():
            action_accuracy = float((decision[mask] == true_dir[mask]).mean())
    elif correct_col is not None:
        mask = ~is_hold
        if mask.any():
            action_accuracy = float(df.loc[mask, correct_col].astype(float).mean())

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    time_x, time_label = _time_axis(df)

    plot_series_with_band(
        time_x,
        p_up,
        window=rolling_window,
        title="Decision Rule P(UP) Over Time",
        xlabel=time_label,
        ylabel="Probability",
        label="p_up",
        out_base=fig_dir / "decision_p_up_time",
        formats=formats,
    )
    plot_series_with_band(
        time_x,
        p_down,
        window=rolling_window,
        title="Decision Rule P(DOWN) Over Time",
        xlabel=time_label,
        ylabel="Probability",
        label="p_down",
        out_base=fig_dir / "decision_p_down_time",
        formats=formats,
    )

    if scan_thresholds:
        thresholds = np.arange(t_min, t_max + 1e-9, t_step)
        hold_rates = []
        conflict_rates = []
        accuracies = []
        true_dir = None
        if true_col is not None:
            true_dir = _normalize_direction(df[true_col])

        for t in thresholds:
            decisions = _decision_rule_from_probs(p_up, p_down, t)
            decisions = pd.Series(decisions)
            holds = decisions.eq("HOLD")
            hold_rates.append(float(holds.mean()))
            conflict_rates.append(float(np.mean((p_up >= t) & (p_down >= t))))
            if true_dir is not None:
                mask = ~holds
                if mask.any():
                    accuracies.append(float((decisions[mask] == true_dir[mask]).mean()))
                else:
                    accuracies.append(np.nan)
            else:
                accuracies.append(np.nan)

        plot_series_with_band(
            thresholds,
            hold_rates,
            window=max(2, min(rolling_window, len(thresholds))),
            title="Hold Rate vs Threshold",
            xlabel="Threshold",
            ylabel="Hold rate",
            label="hold_rate",
            out_base=fig_dir / "decision_hold_rate_threshold",
            formats=formats,
        )
        plot_series_with_band(
            thresholds,
            conflict_rates,
            window=max(2, min(rolling_window, len(thresholds))),
            title="Conflict Rate vs Threshold",
            xlabel="Threshold",
            ylabel="Conflict rate",
            label="conflict_rate",
            out_base=fig_dir / "decision_conflict_rate_threshold",
            formats=formats,
        )
        plot_series_with_band(
            thresholds,
            accuracies,
            window=max(2, min(rolling_window, len(thresholds))),
            title="Action Accuracy vs Threshold",
            xlabel="Threshold",
            ylabel="Accuracy (non-hold)",
            label="accuracy_non_hold",
            out_base=fig_dir / "decision_accuracy_threshold",
            formats=formats,
        )

    return {
        "model": "decision_rule",
        "n": int(len(df)),
        "threshold": float(threshold),
        "hold_rate": hold_rate,
        "action_rate": action_rate,
        "conflict_rate": conflict_rate,
        "action_accuracy_non_hold": float(action_accuracy),
    }


def _extract_hold_mask(df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
    is_hold_col = infer_column(df, ["is_hold", "hold", "is_hold_flag"], "is_hold", required=False)
    action_col = infer_column(df, ["action", "decision", "signal"], "action", required=False)
    if is_hold_col is not None:
        return df[is_hold_col].astype(float) > 0.5, df[action_col] if action_col else None
    if action_col is None:
        raise ValueError(
            f"Missing action/is_hold columns. Columns={list(df.columns)}"
        )
    action = df[action_col]
    if np.issubdtype(action.dtype, np.number):
        return action.astype(int) == 1, action
    action_str = action.astype(str).str.upper()
    return action_str.isin(["HOLD", "FLAT"]), action


def _map_labels_to_steps(step_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    if "index" not in step_df.columns:
        return pd.Series([np.nan] * len(step_df))
    idx = step_df["index"].astype(int)
    valid = (idx >= 0) & (idx < len(labels))
    mapped = pd.Series([np.nan] * len(step_df))
    mapped.loc[valid] = labels[idx[valid]]
    return mapped


def analyze_rl(
    name: str,
    step_df: pd.DataFrame,
    episode_df: Optional[pd.DataFrame],
    out_dir: Path,
    formats: Sequence[str],
    rolling_window: int,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Analyze RL logs and generate plots."""
    p_dir_col = infer_column(step_df, ["p_direction", "prob", "p_dir"], "p_direction", required=False)
    correct_col = infer_column(step_df, ["correct", "is_correct"], "correct", required=False)
    hold_mask, _ = _extract_hold_mask(step_df)

    action_mask = ~hold_mask
    hold_rate = float(hold_mask.mean())
    action_rate = float(action_mask.mean())

    action_precision = np.nan
    if correct_col is not None and action_mask.any():
        action_precision = float(step_df.loc[action_mask, correct_col].astype(float).mean())

    action_recall = np.nan
    f1 = np.nan
    if labels is not None:
        mapped = _map_labels_to_steps(step_df, labels)
        valid = mapped.notna()
        if valid.any():
            y_true = mapped[valid].astype(int)
            y_pred = action_mask[valid].astype(int)
            precision, recall, f1_val, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
            action_precision = float(precision)
            action_recall = float(recall)
            f1 = float(f1_val)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if p_dir_col is not None:
        time_x, time_label = _time_axis(step_df)
        plot_series_with_band(
            time_x,
            step_df[p_dir_col].astype(float).to_numpy(),
            window=rolling_window,
            title=f"{name} P(Direction) Over Time",
            xlabel=time_label,
            ylabel="Probability",
            label="p_direction",
            out_base=fig_dir / f"{name.lower()}_p_direction_time",
            formats=formats,
        )

    if episode_df is not None:
        ep = episode_df["episode"].to_numpy()
        for col, title, ylabel in [
            ("reward", "Reward per Episode", "Reward"),
            ("hold_rate", "Hold Rate per Episode", "Hold rate"),
            ("accuracy", "Accuracy (non-hold) per Episode", "Accuracy"),
            ("entropy", "Policy Entropy per Episode", "Entropy"),
        ]:
            if col in episode_df.columns:
                plot_series_with_band(
                    ep,
                    episode_df[col].astype(float).to_numpy(),
                    window=rolling_window,
                    title=f"{name} {title}",
                    xlabel="Episode",
                    ylabel=ylabel,
                    label=col,
                    out_base=fig_dir / f"{name.lower()}_{col}_episode",
                    formats=formats,
                )

    return {
        "model": name,
        "n": int(len(step_df)),
        "hold_rate": hold_rate,
        "action_rate": action_rate,
        "action_precision": float(action_precision),
        "action_recall": float(action_recall),
        "f1": float(f1),
    }


def summarize_metrics(metrics: List[Dict[str, float]], out_dir: Path) -> pd.DataFrame:
    """Save summary metrics as csv."""
    df = pd.DataFrame(metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df


def write_summary_md(
    out_dir: Path,
    summary_df: pd.DataFrame,
    inputs: AnalysisInputs,
    time_range: Optional[Tuple[float, float]] = None,
    notes: Optional[List[str]] = None,
) -> None:
    """Write summary markdown report."""
    table_lines = _markdown_table(summary_df)
    lines: List[str] = []
    lines.append("# Direction Effectiveness Summary")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- run_dir: {inputs.run_dir}")
    lines.append(f"- threshold: {inputs.threshold:.4f}")
    lines.append(f"- rolling_window: {inputs.rolling_window}")
    lines.append(f"- scan_thresholds: {inputs.scan_thresholds}")
    if inputs.scan_thresholds:
        lines.append(
            f"- threshold_range: {inputs.t_min:.2f}..{inputs.t_max:.2f} step {inputs.t_step:.2f}"
        )
    lines.append(f"- output_dir: {inputs.out_dir}")
    if time_range is not None:
        lines.append(f"- time_range_ms: {int(time_range[0])}..{int(time_range[1])}")
    lines.append("")
    lines.append("## Metrics")
    lines.extend(table_lines)
    lines.append("")
    lines.append("## Notes")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No additional notes.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_run(inputs: AnalysisInputs) -> Path:
    """Run full direction effectiveness analysis and generate outputs."""
    logger = get_logger("analysis.direction_effectiveness")
    artifacts = list_run_artifacts(inputs.run_dir)
    metrics_rows: List[Dict[str, float]] = []
    notes: List[str] = []

    cfg = load_config(inputs.config_path)
    y_up, y_down = load_rl_labels(cfg, inputs.run_dir)
    if y_up is None or y_down is None:
        logger.warning("Dataset for RL labels not found; RL recall/f1 may be NaN.")

    time_range = None

    # Supervised UP/DOWN
    for name, key in [("supervised_up", "supervised_up"), ("supervised_down", "supervised_down")]:
        path = artifacts.get(key)
        if path and path.exists():
            df = load_dataframe(path)
            metrics = analyze_supervised(
                name=name,
                df=df,
                threshold=inputs.threshold,
                out_dir=inputs.out_dir,
                formats=inputs.formats,
                rolling_window=inputs.rolling_window,
            )
            metrics_rows.append(metrics)
            if time_range is None:
                time_col = infer_column(df, TIME_CANDIDATES, "time", required=False)
                if time_col is not None:
                    time_vals = df[time_col].astype(float).to_numpy()
                    time_range = (float(np.min(time_vals)), float(np.max(time_vals)))
            notes.append(
                f"{name}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, coverage={metrics['coverage']:.3f}"
            )
        else:
            logger.warning("Missing %s log at %s", name, path)

    # Decision rule
    decision_path = artifacts.get("decision_rule")
    if decision_path and decision_path.exists():
        decision_df = load_dataframe(decision_path)
        decision_metrics = analyze_decision_rule(
            decision_df,
            threshold=inputs.threshold,
            out_dir=inputs.out_dir,
            formats=inputs.formats,
            rolling_window=inputs.rolling_window,
            scan_thresholds=inputs.scan_thresholds,
            t_min=inputs.t_min,
            t_max=inputs.t_max,
            t_step=inputs.t_step,
        )
        metrics_rows.append(decision_metrics)
        notes.append(
            "decision_rule: hold_rate={:.3f}, conflict_rate={:.3f}, accuracy_non_hold={}".format(
                decision_metrics["hold_rate"],
                decision_metrics["conflict_rate"],
                "nan"
                if np.isnan(decision_metrics["action_accuracy_non_hold"])
                else f"{decision_metrics['action_accuracy_non_hold']:.3f}",
            )
        )

    # RL UP/DOWN
    rl_pairs = [
        ("rl_up", "rl_up_step", "rl_up_episode", y_up),
        ("rl_down", "rl_down_step", "rl_down_episode", y_down),
    ]
    for name, step_key, ep_key, labels in rl_pairs:
        step_path = artifacts.get(step_key)
        if not step_path or not step_path.exists():
            logger.warning("Missing %s step log at %s", name, step_path)
            continue
        step_df = load_dataframe(step_path)
        episode_df = None
        ep_path = artifacts.get(ep_key)
        if ep_path and ep_path.exists():
            episode_df = load_dataframe(ep_path)
        metrics = analyze_rl(
            name=name,
            step_df=step_df,
            episode_df=episode_df,
            out_dir=inputs.out_dir,
            formats=inputs.formats,
            rolling_window=inputs.rolling_window,
            labels=labels,
        )
        metrics_rows.append(metrics)
        notes.append(
            f"{name}: action_precision={metrics['action_precision']:.3f}, hold_rate={metrics['hold_rate']:.3f}"
        )

    summary_df = summarize_metrics(metrics_rows, inputs.out_dir)
    write_summary_md(
        inputs.out_dir,
        summary_df,
        inputs=inputs,
        time_range=time_range,
        notes=notes,
    )
    logger.info("Saved summary to %s", inputs.out_dir)
    return inputs.out_dir


def _markdown_table(df: pd.DataFrame) -> List[str]:
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                if np.isnan(val):
                    values.append("nan")
                else:
                    values.append(f"{val:.6g}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return lines
