"""Simple RL environment for directional prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    R_correct: float
    R_wrong: float
    R_opposite: float
    margin_threshold: float
    margin_penalty: float


class DirectionEnv:
    """Contextual bandit style environment over a fixed dataset (UP/DOWN actions)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps_per_episode: int,
        reward: RewardConfig,
        times: np.ndarray | None = None,
        label_action_map: Dict[int, int] | None = None,
        seed: int = 7,
    ) -> None:
        if len(X) == 0:
            raise ValueError("DirectionEnv requires a non-empty dataset.")
        self.X = X
        self.y = y
        self.times = times
        self.full_episode = steps_per_episode <= 0 or steps_per_episode >= len(X)
        self.steps_per_episode = len(X) if self.full_episode else steps_per_episode
        self.reward = reward
        self.label_action_map = label_action_map or {1: 0, -1: 1}
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.steps = 0

    def reset(self) -> np.ndarray:
        if self.full_episode or len(self.X) <= self.steps_per_episode:
            self.idx = 0
        else:
            self.idx = int(self.rng.integers(0, len(self.X) - self.steps_per_episode))
        self.steps = 0
        return self.X[self.idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        curr_idx = self.idx
        label = int(self.y[curr_idx])
        expected_action = self.label_action_map.get(label)
        if expected_action is None:
            reward = -self.reward.R_wrong
            correct = False
        elif action == expected_action:
            reward = self.reward.R_correct
            correct = True
        else:
            reward = -self.reward.R_opposite
            correct = False
        self.idx += 1
        self.steps += 1
        done = self.steps >= self.steps_per_episode or self.idx >= len(self.X)
        next_state = self.X[self.idx] if not done else self.X[self.idx - 1]
        info = {
            "correct": float(correct),
            "index": float(curr_idx),
            "label": float(label),
        }
        if self.times is not None:
            info["time_ms"] = float(self.times[curr_idx])
        return next_state, float(reward), bool(done), info
