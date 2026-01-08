# Direction Effectiveness Summary

## Inputs
- run_dir: runs\2026-01-08_16-36-52
- threshold: 0.5500
- rolling_window: 20
- scan_thresholds: True
- threshold_range: 0.45..0.75 step 0.01
- output_dir: runs\2026-01-08_16-36-52\reports\direction_effectiveness
- time_range_ms: 205200000..239400000

## Metrics
| model | n | threshold | precision | recall | f1 | roc_auc | pr_auc | coverage | base_rate | lift | tn | fp | fn | tp | prob_mean | prob_std | hold_rate | action_rate | conflict_rate | action_accuracy_non_hold | precision_up_system | precision_down_system | action_precision | action_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| supervised_up | 115 | 0.55 | 0 | 0 | 0 | 0.534483 | 0.589881 | 0 | 0.495652 | 0 | 58 | 0 | 57 | 0 | 0.489925 | 0.00978456 | nan | nan | nan | nan | nan | nan | nan | nan |
| supervised_down | 115 | 0.55 | 0 | 0 | 0 | 0.557774 | 0.578544 | 0 | 0.504348 | 0 | 57 | 0 | 58 | 0 | 0.477674 | 0.00409962 | nan | nan | nan | nan | nan | nan | nan | nan |
| decision_rule | 115 | 0.55 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 1 | 0 | 0 | nan | nan | nan | nan | nan |
| rl_up | 150 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.38 | 0.62 | nan | nan | nan | nan | 0.526882 | nan |
| rl_down | 150 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.413333 | 0.586667 | nan | nan | nan | nan | 0.511364 | nan |

## Notes
- supervised_up: precision=0.000, recall=0.000, coverage=0.000
- supervised_down: precision=0.000, recall=0.000, coverage=0.000
- decision_rule: hold_rate=1.000, conflict_rate=0.000, accuracy_non_hold=nan
- rl_up: action_precision=0.527, hold_rate=0.380
- rl_down: action_precision=0.511, hold_rate=0.413