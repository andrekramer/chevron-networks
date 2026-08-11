# Stage 3 independent checkpoint evaluation

Each saved checkpoint was evaluated on 200 fresh, shared episodes. Success requires a positively rewarded terminal interaction; timeout shaping cannot count as success.

| Model | Parameters | Final success | Final return | Curve mean success | Selected-checkpoint success |
|---|---:|---:|---:|---:|---:|
| GRU baseline | 208,455 | 0.033 +/- 0.047 | 0.058 +/- 0.027 | 0.003 +/- 0.004 | 0.000 +/- 0.000 |
| Plain Chevron | 141,831 | 0.000 +/- 0.000 | 0.037 +/- 0.043 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

## Paired differences

- Final Chevron-minus-GRU success: -0.033 +/- 0.047
- Final Chevron wins: 0/3
- Learning-curve Chevron-minus-GRU success: -0.003 +/- 0.004
- Learning-curve Chevron wins: 0/3

This audit reuses existing trained checkpoints. It removes evaluation-set reuse and the permissive success metric, but remains a three-seed result.
