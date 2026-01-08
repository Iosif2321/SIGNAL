from pathlib import Path

import pytest
import torch

from cryptomvp.utils.gpu import CudaUnavailableError

from scripts.test_data_parity import run_parity
from scripts.test_build_dataset import run_build_dataset
from scripts.test_train_baseline import run_supervised
from scripts.test_train_rl import run_rl
from scripts.test_reward_weights import run_reward_weights


def test_pipeline_fast():
    config = "configs/mvp.yaml"
    run_dir = Path("runs/test_fast")

    run_parity(config, fast=True, run_dir=run_dir)
    run_build_dataset(config, fast=True, run_dir=run_dir)

    assert (run_dir / "reports" / "parity" / "summary.md").exists()
    assert (run_dir / "reports" / "dataset" / "summary.md").exists()

    if torch.cuda.is_available():
        run_supervised(config, fast=True, run_dir=run_dir)
        run_rl(config, fast=True, run_dir=run_dir)
        run_reward_weights(config, fast=True, run_dir=run_dir)
        assert (run_dir / "reports" / "supervised_up").exists()
        assert (run_dir / "reports" / "rl_up").exists()
        assert (run_dir / "reports" / "reward_weights").exists()
    else:
        with pytest.raises(CudaUnavailableError):
            run_supervised(config, fast=True, run_dir=run_dir)
        with pytest.raises(CudaUnavailableError):
            run_rl(config, fast=True, run_dir=run_dir)
        with pytest.raises(CudaUnavailableError):
            run_reward_weights(config, fast=True, run_dir=run_dir)
