# Phase 7.1 results: removing auxiliary supervision

The initial Phase 7 run used answer, A-group retrieval, and A/N gate losses.
This comparison removes those auxiliary objectives in stages while holding the
task, architecture, 700 training steps, and five seeds fixed.

## Accuracy

| Supervision | Method | Overall | Matched | No match |
|---|---|---:|---:|---:|
| Full auxiliary | Soft Chevron | 0.9998 ± 0.0002 | 0.9998 ± 0.0003 | 1.0000 ± 0.0000 |
| Retrieval only | Soft Chevron | 0.9998 ± 0.0002 | 0.9997 ± 0.0003 | 1.0000 ± 0.0000 |
| Answer only | Soft Chevron | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Retrieval auxiliary | Joint attention | 0.9976 ± 0.0007 | 0.9967 ± 0.0010 | 1.0000 ± 0.0000 |
| Answer only | Joint attention | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Explicit gate supervision is not required for Soft Chevron to solve this task.
Removing both auxiliary losses slightly improves its measured answer accuracy.
The strong joint-attention baseline also becomes perfect under answer-only
training, so there is no accuracy advantage to claim in this condition.

## Mechanism measurements

| Supervision | Target r | Non-target r | Matched null mass | No-match null mass |
|---|---:|---:|---:|---:|
| Full auxiliary | 0.9210 ± 0.0042 | 0.0055 ± 0.0006 | 0.7215 ± 0.0019 | 0.9944 ± 0.0006 |
| Retrieval only | 0.9062 ± 0.0105 | 0.0186 ± 0.0011 | 0.7170 ± 0.0051 | 0.9814 ± 0.0014 |
| Answer only | 0.9333 ± 0.0042 | 0.0207 ± 0.0010 | 0.8264 ± 0.0023 | 0.9793 ± 0.0013 |

Answer-only Soft Chevron retains the intended mechanism. The matching target is
admitted, non-target templates are strongly suppressed, and unmatched queries
send approximately 98% of their mass to V_null.

Joint attention reaches the same perfect answer accuracy through a different
internal solution. Under answer-only training, its no-match null mass is only
`0.0067 ± 0.0001`. The answer head learns to classify a mixture of unmatched
values as IDK rather than routing the query through the explicit null slot.

This demonstrates an architectural distinction rather than a performance
distinction: Soft Chevron makes retrieval without assent and explicit null
routing the easy solution, while ordinary joint attention is free to implement
the same labels through its downstream classifier.

## Boundary

The gate begins from a sensible Phase 6 calibration (`theta=0.10`, `k=30`).
Answer supervision can preserve and refine that mechanism, but this experiment
does not yet show that it can recover from poor threshold or sharpness
initialization. Initialization and noise sweeps are therefore the next required
test before adding online IDL writes.

