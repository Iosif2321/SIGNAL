"""Run the full MVP pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cryptomvp.utils.run_dir import init_run_dir  # noqa: E402

def run_all(fast: bool, run_dir: str | None) -> Path:
    run_path = init_run_dir(run_dir, "configs/mvp.yaml")
    args = ["--fast"] if fast else []
    scripts = [
        "scripts/test_data_parity.py",
        "scripts/test_build_dataset.py",
        "scripts/test_train_baseline.py",
        "scripts/test_train_rl.py",
        "scripts/test_reward_weights.py",
    ]
    for script in scripts:
        cmd = [
            sys.executable,
            script,
            "--config",
            "configs/mvp.yaml",
            "--run-dir",
            str(run_path),
        ] + args
        subprocess.run(cmd, check=True)
    return run_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_all(args.fast, args.run_dir)
