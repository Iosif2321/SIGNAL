"""RL tuner to select UP/DOWN model parameters per episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from cryptomvp.utils.gpu import require_cuda
from cryptomvp.utils.logging import get_logger


@dataclass
class TunerEpisode:
    episode: int
    candidate_index: int
    candidate_name: str
    reward: float
    entropy: float
    best_score: float
    metrics: Dict[str, float]


@dataclass
class ParamTunerEpisode:
    episode: int
    reward: float
    entropy: float
    best_score: float
    params: Dict[str, Any]
    metrics: Dict[str, float]


def tune_candidates(
    candidates: Sequence[object],
    evaluate_fn: Callable[[object], Dict[str, float]],
    episodes: int,
    entropy_bonus: float,
    seed: int = 7,
    lr: float = 0.1,
) -> Tuple[List[TunerEpisode], torch.Tensor]:
    """Run a REINFORCE-style bandit tuner over candidate configs."""
    device = require_cuda()
    logger = get_logger("rl_tuner")
    torch.manual_seed(seed)
    np.random.seed(seed)

    logits = torch.zeros(len(candidates), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    baseline = 0.0
    best_score = -float("inf")
    history: List[TunerEpisode] = []

    for ep in range(episodes):
        dist = Categorical(logits=logits)
        action = dist.sample()
        entropy = dist.entropy()
        candidate = candidates[int(action.item())]
        metrics = evaluate_fn(candidate)
        reward = float(metrics["score"])
        baseline = 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        loss = -(dist.log_prob(action) * advantage) - entropy_bonus * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if reward > best_score:
            best_score = reward

        history.append(
            TunerEpisode(
                episode=ep + 1,
                candidate_index=int(action.item()),
                candidate_name=str(getattr(candidate, "name", f"cand_{action.item()}")),
                reward=reward,
                entropy=float(entropy.item()),
                best_score=best_score,
                metrics=metrics,
            )
        )

        logger.info(
            "Episode %s/%s - candidate=%s reward=%.4f best=%.4f",
            ep + 1,
            episodes,
            getattr(candidate, "name", action.item()),
            reward,
            best_score,
        )

    return history, logits.detach()


def tune_param_space(
    param_space: Dict[str, List[Any]],
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    episodes: int,
    entropy_bonus: float,
    seed: int = 7,
    lr: float = 0.1,
) -> Tuple[List[ParamTunerEpisode], Dict[str, torch.Tensor]]:
    """Run a factorized REINFORCE tuner over parameter space."""
    device = require_cuda()
    logger = get_logger("rl_tuner")
    torch.manual_seed(seed)
    np.random.seed(seed)

    specs: Dict[str, Dict[str, Any]] = {}
    for name, values in param_space.items():
        if not values:
            raise ValueError(f"Parameter space for {name} is empty.")
        specs[name] = {
            "values": list(values),
            "logits": torch.zeros(len(values), device=device, requires_grad=True),
        }

    optimizer = torch.optim.Adam([spec["logits"] for spec in specs.values()], lr=lr)
    baseline = 0.0
    best_score = -float("inf")
    history: List[ParamTunerEpisode] = []

    for ep in range(episodes):
        log_probs = []
        entropies = []
        params: Dict[str, Any] = {}
        for name, spec in specs.items():
            dist = Categorical(logits=spec["logits"])
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            params[name] = spec["values"][int(action.item())]

        metrics = evaluate_fn(params)
        reward = float(metrics["score"])
        baseline = 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        log_prob_sum = torch.stack(log_probs).sum()
        entropy_mean = torch.stack(entropies).mean()
        loss = -(log_prob_sum * advantage) - entropy_bonus * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if reward > best_score:
            best_score = reward

        history.append(
            ParamTunerEpisode(
                episode=ep + 1,
                reward=reward,
                entropy=float(entropy_mean.item()),
                best_score=best_score,
                params=params,
                metrics=metrics,
            )
        )

        logger.info(
            "Episode %s/%s - reward=%.4f best=%.4f",
            ep + 1,
            episodes,
            reward,
            best_score,
        )

    logits_out = {name: spec["logits"].detach().cpu() for name, spec in specs.items()}
    return history, logits_out
