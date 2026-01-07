# Direction Effectiveness Summary

## Inputs
- run_dir: runs\2026-01-07
- threshold: 0.5500
- rolling_window: 20
- scan_thresholds: True
- threshold_range: 0.45..0.75 step 0.01
- output_dir: runs\2026-01-07\reports\direction_effectiveness
- time_range_ms: 1674808800000..1675209000000

## Metrics
| model | n | threshold | precision | recall | f1 | roc_auc | pr_auc | coverage | tn | fp | fn | tp | prob_mean | prob_std | hold_rate | action_rate | conflict_rate | action_accuracy_non_hold | action_precision | action_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| supervised_up | 1335 | 0.55 | 0.666667 | 0.00292826 | 0.0058309 | 0.499669 | 0.519594 | 0.00224719 | 651 | 1 | 681 | 2 | 0.473671 | 0.034869 | nan | nan | nan | nan | nan | nan |
| supervised_down | 1335 | 0.55 | 0 | 0 | 0 | 0.494801 | 0.481575 | 0 | 687 | 0 | 648 | 0 | 0.435081 | 0.0152608 | nan | nan | nan | nan | nan | nan |
| decision_rule | 1335 | 0.55 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.997753 | 0.00224719 | 0 | 0.666667 | nan | nan |
| rl_up | 10000 | nan | nan | nan | 0.512961 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4898 | 0.5102 | nan | nan | 0.513916 | 0.512009 |
| rl_down | 10000 | nan | nan | nan | 0.553473 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.3293 | 0.6707 | nan | nan | 0.473386 | 0.666177 |

## Notes
- supervised_up: precision=0.667, recall=0.003, coverage=0.002
- supervised_down: precision=0.000, recall=0.000, coverage=0.000
- decision_rule: hold_rate=0.998, conflict_rate=0.000, accuracy_non_hold=0.667
- rl_up: action_precision=0.514, hold_rate=0.490
- rl_down: action_precision=0.473, hold_rate=0.329