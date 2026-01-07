"""Simple RL environment for directional prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    R_correct: float
    R_wrong: float
    R_hold: float


class DirectionEnv:
    """Contextual bandit style environment over a fixed dataset."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps_per_episode: int,
        reward: RewardConfig,
        times: np.ndarray | None = None,
        seed: int = 7,
    ) -> None:
        self.X = X
        self.y = y
        self.times = times
        self.steps_per_episode = steps_per_episode
        self.reward = reward
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.steps = 0

    def reset(self) -> np.ndarray:
        if len(self.X) <= self.steps_per_episode:
            self.idx = 0
        else:
            self.idx = int(self.rng.integers(0, len(self.X) - self.steps_per_episode))
        self.steps = 0
        return self.X[self.idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        curr_idx = self.idx
        label = int(self.y[curr_idx])
        if action == 0:  # direction
            reward = self.reward.R_correct if label == 1 else -self.reward.R_wrong
            is_hold = False
            correct = label == 1
        else:
            reward = -self.reward.R_hold
            is_hold = True
            correct = False

        self.idx += 1
        self.steps += 1
        done = self.steps >= self.steps_per_episode or self.idx >= len(self.X)
        next_state = self.X[self.idx] if not done else self.X[self.idx - 1]
        info = {
            "is_hold": float(is_hold),
            "correct": float(correct),
            "index": float(curr_idx),
        }
        if self.times is not None:
            info["time_ms"] = float(self.times[curr_idx])
        return next_state, float(reward), bool(done), info
