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
    low_margin_rates: List[float]
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
    device: torch.device | None = None,
    label_action_map: Dict[int, int] | None = None,
    init_state: Dict[str, torch.Tensor] | None = None,
) -> Tuple[PolicyNet, RLHistory]:
    """Train a policy network with REINFORCE on GPU-only."""
    device = device or require_cuda()
    logger = get_logger(f"rl.{model_name}")

    X = X.astype(np.float32)
    policy = PolicyNet(input_dim=X.shape[1], hidden_dim=policy_hidden_dim).to(device)
    if init_state is not None:
        policy.load_state_dict(init_state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    env = DirectionEnv(
        X,
        y,
        steps_per_episode=steps_per_episode,
        reward=reward_cfg,
        times=times,
        label_action_map=label_action_map,
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
        low_margins = []
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
            margin = float(abs(probs[0].item() - probs[1].item()))
            penalty = 0.0
            if reward_cfg.margin_threshold > 0.0 and margin < reward_cfg.margin_threshold:
                penalty = reward_cfg.margin_penalty * (
                    (reward_cfg.margin_threshold - margin) / reward_cfg.margin_threshold
                )
                reward -= penalty
            rewards.append(reward)
            low_margins.append(float(margin < reward_cfg.margin_threshold))
            corrects.append(info["correct"])
            if track_steps:
                step_logs.append(
                    {
                        "episode": int(ep + 1),
                        "step": int(step_idx + 1),
                        "index": int(info.get("index", -1)),
                        "time_ms": int(info.get("time_ms", -1)),
                        "action": int(action.item()),
                        "p_up": float(probs[0].item()),
                        "p_down": float(probs[1].item()),
                        "margin": margin,
                        "low_margin_penalty": float(penalty),
                        "reward": float(reward),
                        "low_margin": float(margin < reward_cfg.margin_threshold),
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
        low_margin_rate = float(np.mean(low_margins)) if low_margins else 0.0
        acc = float(np.mean(corrects)) if corrects else 0.0
        entropy = float(entropy_t.mean().item()) if entropies else 0.0

        history.rewards.append(avg_reward)
        history.low_margin_rates.append(low_margin_rate)
        history.accuracies.append(acc)
        history.entropies.append(entropy)
        history.grad_norms.append(gnorm)
        history.delta_weight_norms.append(dnorm)
        history.weight_norms.append(weight_norm)

        logger.info(
            "Episode %s/%s - reward=%.4f low_margin=%.3f acc=%.3f entropy=%.3f",
            ep + 1,
            episodes,
            avg_reward,
            low_margin_rate,
            acc,
            entropy,
        )

    history.step_logs = step_logs if track_steps else None
    return policy, history
