"""REINFORCE training for direction policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from cryptomvp.train.diagnostics import grad_norm, weight_delta_norm, params_vector
from cryptomvp.train.rl_env import DirectionEnv, RewardConfig
from cryptomvp.train.rl_policy import PolicyNet
from cryptomvp.utils.gpu import require_cuda
from cryptomvp.utils.logging import get_logger


@dataclass
class RLHistory:
    rewards: List[float]
    hold_rates: List[float]
    accuracies: List[float]
    entropies: List[float]
    grad_norms: List[float]
    delta_weight_norms: List[float]
    weight_norms: List[float]
    step_logs: List[Dict[str, object]] | None


def _discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def train_reinforce(
    X: np.ndarray,
    y: np.ndarray,
    episodes: int,
    steps_per_episode: int,
    lr: float,
    gamma: float,
    reward_cfg: RewardConfig,
    policy_hidden_dim: int = 64,
    entropy_bonus: float = 0.0,
    seed: int = 7,
    track_diagnostics: bool = False,
    track_steps: bool = False,
    times: np.ndarray | None = None,
    model_name: str = "policy",
) -> Tuple[PolicyNet, RLHistory]:
    """Train a policy network with REINFORCE on GPU-only."""
    device = require_cuda()
    logger = get_logger(f"rl.{model_name}")

    X = X.astype(np.float32)
    policy = PolicyNet(input_dim=X.shape[1], hidden_dim=policy_hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    env = DirectionEnv(
        X,
        y,
        steps_per_episode=steps_per_episode,
        reward=reward_cfg,
        times=times,
        seed=seed,
    )

    history = RLHistory([], [], [], [], [], [], [], None)
    baseline = 0.0
    step_logs: List[Dict[str, object]] = []

    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        entropies = []
        holds = []
        corrects = []

        for step_idx in range(steps_per_episode):
            state_t = torch.from_numpy(state).float().to(device).unsqueeze(0)
            logits = policy(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            probs = torch.softmax(logits, dim=1).squeeze(0)

            next_state, reward, done, info = env.step(int(action.item()))
            rewards.append(reward)
            holds.append(info["is_hold"])
            if info["is_hold"] < 0.5:
                corrects.append(info["correct"])
            if track_steps:
                step_logs.append(
                    {
                        "episode": int(ep + 1),
                        "step": int(step_idx + 1),
                        "index": int(info.get("index", -1)),
                        "time_ms": int(info.get("time_ms", -1)),
                        "action": int(action.item()),
                        "p_direction": float(probs[0].item()),
                        "p_hold": float(probs[1].item()),
                        "reward": float(reward),
                        "is_hold": float(info["is_hold"]),
                        "correct": float(info["correct"]),
                        "label": float(info.get("label", 0.0)),
                        "state": state.astype(np.float32).tolist(),
                    }
                )
            state = next_state
            if done:
                break

        returns = _discounted_returns(rewards, gamma)
        returns_t = torch.tensor(returns, device=device)
        baseline = 0.9 * baseline + 0.1 * float(returns_t.mean().item())
        advantages = returns_t - baseline

        log_probs_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies)
        loss = -(log_probs_t * advantages.detach()).mean() - entropy_bonus * entropy_t.mean()

        optimizer.zero_grad()
        loss.backward()
        gnorm = grad_norm(policy) if track_diagnostics else 0.0
        prev_params = params_vector(policy).detach().clone() if track_diagnostics else None
        optimizer.step()
        dnorm = weight_delta_norm(prev_params, policy) if track_diagnostics else 0.0
        weight_norm = float(torch.norm(params_vector(policy), p=2).item())

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        hold_rate = float(np.mean(holds)) if holds else 0.0
        acc = float(np.mean(corrects)) if corrects else 0.0
        entropy = float(entropy_t.mean().item()) if entropies else 0.0

        history.rewards.append(avg_reward)
        history.hold_rates.append(hold_rate)
        history.accuracies.append(acc)
        history.entropies.append(entropy)
        history.grad_norms.append(gnorm)
        history.delta_weight_norms.append(dnorm)
        history.weight_norms.append(weight_norm)

        logger.info(
            "Episode %s/%s - reward=%.4f hold=%.3f acc=%.3f entropy=%.3f",
            ep + 1,
            episodes,
            avg_reward,
            hold_rate,
            acc,
            entropy,
        )

    history.step_logs = step_logs if track_steps else None
    return policy, history
