"""Run the full MVP pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_all(fast: bool) -> None:
    args = ["--fast"] if fast else []
    scripts = [
        "scripts/test_data_parity.py",
        "scripts/test_build_dataset.py",
        "scripts/test_train_baseline.py",
        "scripts/test_train_rl.py",
        "scripts/test_reward_weights.py",
    ]
    for script in scripts:
        cmd = [sys.executable, script, "--config", "configs/mvp.yaml"] + args
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run_all(args.fast)