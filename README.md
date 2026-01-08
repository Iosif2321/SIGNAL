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

### Test 5: Reward/penalty diagnostics
```bash
python scripts/test_reward_weights.py --config configs/mvp.yaml
```

### Run all
```bash
python scripts/mvp_run_all.py --fast
```

## Reports

Artifacts are stored under `runs/<run_id>/reports/`:
- `runs/<run_id>/reports/parity/`
- `runs/<run_id>/reports/dataset/`
- `runs/<run_id>/reports/supervised_up/`, `runs/<run_id>/reports/supervised_down/`
- `runs/<run_id>/reports/rl_up/`, `runs/<run_id>/reports/rl_down/`
- `runs/<run_id>/reports/reward_weights/`
- `runs/<run_id>/reports/decision_rule/`

Run metadata and copied config live in `runs/<run_id>/metadata.json` and `runs/<run_id>/config.yaml`.

## Pytest

```bash
pytest -q
```

## Notes

- Training/evaluation is GPU-only. If CUDA is not available, training/evaluation will raise a clear error.
- The config is in `configs/mvp.yaml`.
