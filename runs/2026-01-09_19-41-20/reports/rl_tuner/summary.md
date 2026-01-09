# RL Tuner Summary (Full Pipeline)
Symbol: BTCUSDT
Interval: 5
Seed: 42
Episodes: 50
Param space entries: 24
Search explore_prob: 0.1
Search mutate_prob: 0.5
Search max_mutations: 2
Search neighbor_only: False
Search always_mutate: True
Search method: best_anchor
Decision weights: acc=1.0, action=0.5, conflict_penalty=0.7, hold_penalty=0.7
RL weights: up_acc=0.2, down_acc=0.4, up_hold_penalty=0.1, down_hold_penalty=0.2
Improve weights: up=0.4, down=0.4
Best reward: 1.2654
Adaptation good rate: 0.1200
Best params:
{
  "features.list_of_features": [
    "returns",
    "log_returns",
    "volatility",
    "range",
    "volume",
    "rsi_14",
    "ema_diff"
  ],
  "features.window_size_K": 32,
  "supervised.epochs": 50,
  "supervised.batch_size": 64,
  "supervised.lr": 0.0005,
  "supervised.weight_decay": 0.0001,
  "supervised.hidden_dim": 64,
  "supervised.early_stopping_patience": 10,
  "decision_rule.T_min": 0.6,
  "decision_rule.delta_min": 0.02,
  "decision_rule.use_best_from_scan": true,
  "decision_rule.scan_min": 0.45,
  "decision_rule.scan_max": 0.75,
  "decision_rule.scan_step": 0.01,
  "rl.episodes": 50,
  "rl.steps_per_episode": 200,
  "rl.gamma": 0.99,
  "rl.lr": 0.0005,
  "rl.entropy_bonus": 0.02,
  "rl.reward.R_correct": 1.0,
  "rl.reward.R_wrong": 1.0,
  "rl.reward.R_opposite": 2.0,
  "rl.reward.R_hold": 0.4,
  "rl.policy_hidden_dim": 128
}