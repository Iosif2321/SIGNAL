# Crypto MVP (BTCUSDT Direction)

MVP for BTCUSDT (Bybit spot) direction prediction with separate UP/DOWN models, GPU-only training/evaluation, and mandatory visualization artifacts.

## Setup

1) Create a Python 3.11+ environment.
2) Install requirements:

```bash
pip install -r requirements.txt
```

3) Install PyTorch with CUDA support by following the official PyTorch installation instructions for your system.

## Run Tests / Scripts

All scripts create a run folder under `runs/` (timestamped) with:
`runs/<run_id>/reports/`, `runs/<run_id>/data/`, and `runs/<run_id>/checkpoints/`.
You can override the run folder with `--run-dir`.

### Test 1: WS vs REST parity
```bash
python scripts/test_data_parity.py --config configs/mvp.yaml
```

### Test 2: Build + validate dataset
```bash
python scripts/test_build_dataset.py --config configs/mvp.yaml
```

### Test 3: Supervised baseline (UP/DOWN)
```bash
python scripts/test_train_baseline.py --config configs/mvp.yaml
```

### Test 4: RL training (UP/DOWN)
```bash
python scripts/test_train_rl.py --config configs/mvp.yaml
```

### Test 5: RL tuner (full pipeline)
```bash
python scripts/test_rl_tuner.py --config configs/mvp.yaml
```

### Test 6: Reward/penalty diagnostics
```bash
python scripts/test_reward_weights.py --config configs/mvp.yaml
```

### Run all (real data, month window)
```bash
# Full training on the fixed month window:
python scripts/mvp_run_all.py

# Faster run (same real dataset, fewer epochs/episodes):
python scripts/mvp_run_all.py --fast
```

## Reports

Artifacts are stored under `runs/<run_id>/reports/`:
- `runs/<run_id>/reports/parity/`
- `runs/<run_id>/reports/dataset/`
- `runs/<run_id>/reports/supervised_up/`, `runs/<run_id>/reports/supervised_down/`
- `runs/<run_id>/reports/rl_up/`, `runs/<run_id>/reports/rl_down/`
- `runs/<run_id>/reports/rl_tuner/`
- `runs/<run_id>/reports/reward_weights/`
- `runs/<run_id>/reports/decision_rule/`

Run metadata and copied config live in `runs/<run_id>/metadata.json` and `runs/<run_id>/config.yaml`.

## Real Data Requirement

All pipeline scripts expect the fixed month dataset defined in `configs/mvp.yaml`.
Run `scripts/test_build_dataset.py` first to download and cache the dataset under the run folder.

## Decision Rule Notes

- `decision_rule.T_min` is the minimum confidence threshold.
- `decision_rule.delta_min` enforces a minimum separation between `p_up` and `p_down` to avoid conflicts.
- `decision_rule.delta_grid` is used to scan thresholds and pick the best (saved under `reports/decision_rule/`).
- `decision_rule.use_best_from_scan` controls whether decision logs use the best scan result or the fixed config value.

## RL Tuner Notes

- The tuner supports two modes via `tuner.mode` in `configs/mvp.yaml`:
  - `agent`: trains RL UP/DOWN policies directly across tuner episodes and evaluates them on a validation split.
  - `search`: runs a **full pipeline** per episode (supervised + RL + decision_rule) over a parameter search space.
- For `agent` mode, `tuner.agent_train_episodes` and `tuner.agent_steps_per_episode` control how much RL training is done per tuner episode.
- For `search` mode, the parameter search space is defined under `tuner.param_space` in `configs/mvp.yaml` and uses dotted keys (e.g. `supervised.lr`, `rl.reward.R_opposite`, `features.list_of_features`).
- In `search` mode, each episode starts from scratch (new UP/DOWN models and new RL policies).

## Pytest

```bash
pytest -q
```

## Notes

- Training/evaluation is GPU-only. If CUDA is not available, training/evaluation will raise a clear error.
- The config is in `configs/mvp.yaml`.
