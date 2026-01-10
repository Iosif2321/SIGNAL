# Pipeline Map

This document maps the current pipeline and where each stage lives in the repo.

## Architecture (current)

Data (Bybit OHLCV) -> Features -> Windows -> Supervised UP/DOWN -> RL UP/DOWN -> Decision Rule -> Reports

### Data + Labels
- REST client: `src/cryptomvp/bybit/rest.py`
- WS client: `src/cryptomvp/bybit/ws.py`
- Dataset build: `src/cryptomvp/data/build_dataset.py`
- Dataset validation: `src/cryptomvp/data/validate_dataset.py`
- Feature computation: `src/cryptomvp/data/features.py`
- Feature registry: `features/registry.yaml`
- Feature sets: `features/feature_sets.yaml`
- Staged selection: `src/cryptomvp/features/selection.py` + `scripts/feature_selection.py`
- Windowing: `src/cryptomvp/data/windowing.py`
- Labels:
  - Supervised UP/DOWN (1-bar horizon): `src/cryptomvp/data/labels.py::make_up_down_labels`
  - RL directional (+1/-1/0): `src/cryptomvp/data/labels.py::make_directional_labels`

### Supervised (UP/DOWN)
- Models: `src/cryptomvp/models/up_model.py`, `src/cryptomvp/models/down_model.py`
- Training loop: `src/cryptomvp/train/supervised.py`
- Evaluation metrics: `src/cryptomvp/train/eval_supervised.py`
- Script: `scripts/test_train_baseline.py`

### RL (UP/DOWN)
- Env + reward: `src/cryptomvp/train/rl_env.py`
- Policy network: `src/cryptomvp/train/rl_policy.py`
- Training loop: `src/cryptomvp/train/rl_train.py`
- Script: `scripts/test_train_rl.py`

### Decision Rule
- Logic (UP/DOWN/HOLD): `src/cryptomvp/decision/rule.py`
- Threshold scan (T_min / delta_min): `scripts/test_train_baseline.py`

### Fixed-Period Evaluation
- CLI: `python -m cryptomvp.evaluate_fixed_period`
- Module: `src/cryptomvp/evaluate_fixed_period.py`

### Walk-Forward Evaluation
- CLI: `python -m cryptomvp.walk_forward`
- Module: `src/cryptomvp/walk_forward.py`

### RL Tuner (full pipeline search)
- Orchestrator: `scripts/test_rl_tuner.py`
- Legacy bandit tuner: `src/cryptomvp/train/rl_tuner.py`

### Monitoring / Adaptation
- Rolling metrics + drift: `scripts/monitoring_report.py`
- Criteria check: `src/cryptomvp/analysis/adaptation.py`

### Artifacts / Runs
- Run directory setup: `src/cryptomvp/utils/run_dir.py`
- Output paths: `src/cryptomvp/utils/io.py`
- Reports saved under: `runs/<run_id>/reports/...`

## Defaults (from `configs/mvp.yaml`)
- Symbol: BTCUSDT, Category: spot, Interval: 5 (minutes)
- Fixed dataset period: `dataset.start_date` .. `dataset.end_date`
- Output path: `dataset.output_path`

## Label Definition (ground truth)
- Horizon: 1 bar ahead.
- Supervised UP/DOWN: compare `close[t+1]` vs `close[t]`.
  - UP if `close[t+1] > close[t]`, DOWN if `close[t+1] < close[t]`.
  - FLAT when equal (both labels 0).
- RL directional labels:
  - UP model: +1 (up), -1 (down), 0 (flat)
  - DOWN model: +1 (down), -1 (up), 0 (flat)

## Timestamps / Timezone
- `open_time_ms` comes from Bybit REST/WS and is treated as UTC milliseconds.
- No timezone conversion is currently applied in core pipeline.

## Commands (current)

Build + validate dataset:
```
python scripts/test_build_dataset.py --config configs/mvp.yaml
```

Supervised UP/DOWN:
```
python scripts/test_train_baseline.py --config configs/mvp.yaml
```

RL UP/DOWN:
```
python scripts/test_train_rl.py --config configs/mvp.yaml
```

RL Tuner:
```
python scripts/test_rl_tuner.py --config configs/mvp.yaml
```

Full run:
```
python scripts/mvp_run_all.py --fast
```

Fixed-period evaluation (overall + per-session):
```
python -m cryptomvp.evaluate_fixed_period --config configs/mvp.yaml
```

Walk-forward evaluation:
```
python -m cryptomvp.walk_forward --config configs/mvp.yaml
```

Feature selection:
```
python scripts/feature_selection.py --config configs/mvp.yaml --top-n 10
```

Monitoring report:
```
python scripts/monitoring_report.py --config configs/mvp.yaml --run-dir runs/<run_id>
```

## Session Layer (implemented)
- Session router and overrides: `src/cryptomvp/sessions/`
- Session annotations: `assign_session_features` in `src/cryptomvp/sessions/router.py`
- Per-session artifacts will be placed under:
  - `runs/<run_id>/sessions/ASIA/`
  - `runs/<run_id>/sessions/EUROPE/`
  - `runs/<run_id>/sessions/US/`
