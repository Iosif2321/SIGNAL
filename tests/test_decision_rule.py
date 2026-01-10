import numpy as np

from cryptomvp.decision.rule import batch_decide, decide, hold_rate


def test_decision_rule_thresholds():
    assert decide(0.7, 0.2, t_min=0.6) == "UP"
    assert decide(0.2, 0.8, t_min=0.6) == "DOWN"
    assert decide(0.55, 0.52, t_min=0.6) == "HOLD"
    assert decide(0.55, 0.52, t_min=0.5, delta_min=0.05) == "HOLD"
    assert decide(0.60, 0.52, t_min=0.5, delta_min=0.05) == "UP"


def test_batch_decide_hold_rate():
    p_up = np.array([0.6, 0.4, 0.7])
    p_down = np.array([0.3, 0.6, 0.2])
    decisions = batch_decide(p_up, p_down, t_min=0.5)
    assert decisions == ["UP", "DOWN", "UP"]
    assert hold_rate(decisions) == 0.0
