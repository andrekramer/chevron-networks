# Phase 6 final results

Phase 6 asked whether category formation and ART-like matching supplied the
missing stability/plasticity mechanism for Chevron Attention. Three controlled
experiments tested persistent category allocation, recurring category drift,
and contextual ambiguity. The final robustness suite repeated the comparisons
over 20 paired seeds while varying duration, noise, separation, capacity,
shift size, vigilance, and distractor difficulty.

All methods saw identical streams or episodes within a seed. Default
hyperparameters were frozen rather than retuned for each condition. Reported
intervals are normal 95% confidence intervals across seeds.

## 1. Persistent category allocation

The default stream contained three recurring categories, eight coherent but
brief categories, and one sustained new category. Four retained categories are
therefore sufficient only if brief categories are not consolidated.

| Condition | Method | Old accuracy after brief categories | Final accuracy | Brief categories retained |
|---|---|---:|---:|---:|
| Default | Chevron ART-like | 1.000 | 1.000 | 0 |
| Default | Persistent single-template attention | 1.000 | 1.000 | 0 |
| Default | Immediate-write attention | 0.333 | 0.250 | 8 |
| Noise 0.06 | Chevron ART-like | 1.000 | 1.000 | 0 |
| Noise 0.06 | Persistent single-template attention | 1.000 | 1.000 | 0 |
| Noise 0.08 | Chevron ART-like | 0.083 ± 0.080 | 0.166 ± 0.067 | 0 |
| Noise 0.08 | Persistent single-template attention | 0.083 ± 0.080 | 0.166 ± 0.067 | 0 |

The category-creation threshold also produced a sharp duration boundary.
Transients lasting two through six observations were rejected on all 20 seeds.
At seven observations they crossed the fixed persistence threshold and were
consolidated. The four-slot Chevron memory then retained four of the eight
brief categories and final accuracy fell to `0.25`.

This is the intended temporal decision, but it is a calibrated decision—not an
intrinsically discovered distinction between temporary and permanent events.

The important negative result persists across the sweep: single-template
persistent attention ties the Chevron A/N memory whenever both are given enough
category capacity. Category creation alone does not establish an A/N advantage.

## 2. Recurring category drift

One learned category moved temporarily, returned to its retained form, and
later made the same move persistently. This requires responsiveness to the
current surface and recovery of the retained surface.

| Condition | Method | Minimum current/recovered accuracy | Long-shift online accuracy | Adaptation steps | Retained final accuracy |
|---|---|---:|---:|---:|---:|
| Default | Soft-contrast Chevron | 1.000 | 0.986 | 1.00 | 1.000 |
| Default | Standard dual-trace attention | 1.000 | 0.986 | 1.00 | 1.000 |
| Default | Hard-search Chevron | 1.000 | 0.942 ± 0.001 | 4.05 ± 0.10 | 1.000 |
| Default | Persistent single template | 0.000 | 0.844 ± 0.006 | 10.95 ± 0.39 | 1.000 |
| Noise 0.07 | Soft-contrast Chevron | 0.988 ± 0.010 | 0.986 ± 0.003 | 1.00 ± 0.20 | 1.000 |
| Noise 0.07 | Standard dual-trace attention | 0.988 ± 0.010 | 0.986 ± 0.003 | 1.00 ± 0.20 | 1.000 |
| Noise 0.07 | Hard-search Chevron | 0.756 ± 0.106 | 0.863 ± 0.015 | 7.15 ± 1.84 | 1.000 |

Dual traces solve the tested fast-current/slow-retained tradeoff much more
cleanly than one persistence-gated template. But standard attention over the
same fast and retained tokens ties soft Chevron throughout the duration,
noise, displacement, and IDL-threshold sweeps.

Hard vigilance/search is consistently slower and becomes substantially less
reliable as noise rises.

## 3. Contextual ambiguity

In each episode, bottom-up decoys outranked the target on A similarity. Only
the target N template matched the top-down context. All decoys carried the same
wrong value, allowing their softmax mass to accumulate.

| Condition | Joint softmax | Joint top-1 | Soft vigilance | Sharp vigilance | Masked attention | Hard search |
|---|---:|---:|---:|---:|---:|---:|
| 31 decoys | 0.920 ± 0.044 | 1.000 | 0.920 ± 0.052 | 1.000 | 1.000 | 1.000 |
| 63 decoys | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| Target A noise 0.12 | 0.400 ± 0.099 | 1.000 | 0.480 ± 0.108 | 1.000 | 1.000 | 1.000 |
| Template noise 0.07 | 0.570 ± 0.107 | 1.000 | 0.990 ± 0.020 | 1.000 | 0.690 ± 0.092 | 0.690 ± 0.092 |

Hard search is robust to distractor multiplicity: it can reject 63
higher-ranked memories. But its fixed threshold becomes brittle when the true
template itself is noisy. Joint top-1 and sharp differentiable matching retain
perfect accuracy in that condition.

Soft vigilance is also calibrated. At the default 31-decoy condition, reducing
sharpness to `40` causes complete failure, while sharpness `120` gives perfect
accuracy. Raising the vigilance threshold from `0.06` to `0.08` makes the
default soft gate fail, although its sharper version remains perfect.

Complementarity alone is not a mismatch detector. In the base ambiguity task,
target complementarity averaged `0.9997`, but deliberately conflicting decoys
still averaged `0.9496`. The complementarity-only gate therefore failed almost
completely.

## What survived Phase 6

The experiments support four architectural ingredients:

1. separate fast and retained traces;
2. persistence-controlled writes and category allocation;
3. normalized absolute A/N contrast as a mismatch signal;
4. soft or top-1 conditional retrieval using both A and N.

They do not support three stronger claims:

1. Full ART reset/search is not superior to modern attention here.
2. Fixed vigilance is not robust across observation and template noise.
3. Chevron routing has not outperformed standard attention given the same two
   traces and conditional gate.

The strongest defensible Phase 6 claim is:

> Across controlled category-allocation, drift, and contextual-ambiguity tasks,
> paired fast and retained traces with persistence-controlled writes achieve a
> useful stability/plasticity tradeoff. Normalized A/N matching prevents
> misleading bottom-up memories from controlling retrieval. These benefits do
> not require full ART reset/search and are equally expressible by suitably
> structured standard attention in the tested regimes.

## Architectural conclusion

The Phase 7 candidate should therefore be ART-inspired rather than an ART
implementation:

```text
Q_A, K_A                 fast retrieval
V_N                      retained content
M(A,N)                   normalized mismatch
soft learned admission   current conditional influence
IDL                      persistence-controlled N writes
null value               retrieval without assent
```

The vigilance threshold should become learned, normalized, or adaptive rather
than remain a fixed hand-tuned constant. Hard search should remain an ablation,
not the default path.

A no-replay MLP was intentionally excluded from the final headline comparison
because it is not competitive or memory matched. A replay-equipped MLP and
learned Q/K projections belong in the end-to-end learned-network phase.

