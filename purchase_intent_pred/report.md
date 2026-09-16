# Purchase-intent prediction: final-setting report

## Experimental setting

The final experiment uses a multi-store commerce dataset, action-only histories,
a 500-session imbalanced real test set, Qwen3.5-4B, and greedy free generation.
The test set contains 45 purchase and 455 terminate sessions. Training contains
739 paired IDs in both TRTR and TSTR; TRSTR contains the real and synthetic
trajectory for each ID.

The model generates exactly `{"action":"purchase"}` or
`{"action":"terminate"}`. No candidate-logit scoring, threshold calibration,
or validation-selected operating point is used. F1 for purchase is the primary
metric.

## Preliminary three-seed result

The current completed run includes seeds 13, 17, and 23 for TRTR and TSTR:

| Condition | Mean F1 | Sample SD | Seeds |
|---|---:|---:|---:|
| TRTR | 0.6436 | 0.0147 | 3 |
| TSTR | 0.6552 | 0.0143 | 3 |
| TSTR − TRTR | +0.0116 | 0.0147 | 3 paired differences |


## Statistical interpretation

For the three paired seeds, the conventional paired test of zero mean
TSTR-minus-TRTR difference gives `p = 0.305`. This means the experiment does not
detect a difference; it does not prove equivalence. The approximate paired 95%
confidence interval is `[-0.025, 0.048]`, which still contains potentially
meaningful effects.
