# Phase 7 initial results: learned soft Chevron Attention

Five independently initialized models per method (`7`, `17`, `27`, `37`, and
`47`) were trained for 700 steps. Each model was evaluated on 2,560 fresh
episodic memories, giving 12,800 held-out examples per method.

Each memory contained three A-address groups with three category members per
group. Q_A/K_A could identify the relevant group, but A alone could not identify
the correct member. Current match evidence agreed with exactly one retained N
template in 75% of examples. In the remaining 25%, no N template matched and
the correct answer was IDK.

Values and category memories were regenerated for every batch.

## Held-out result

| Method | Parameters | Overall accuracy | Matched accuracy | No-match accuracy | A-group retrieval |
|---|---:|---:|---:|---:|---:|
| Soft Chevron | 3,759 | 0.9998 ± 0.0002 | 0.9998 ± 0.0003 | 1.0000 ± 0.0000 | 0.9959 ± 0.0010 |
| Standard joint A/N attention | 4,254 | 0.9976 ± 0.0007 | 0.9967 ± 0.0010 | 1.0000 ± 0.0000 | 0.8948 ± 0.0045 |
| A-only attention | 3,614 | 0.2570 ± 0.0110 | 0.2226 ± 0.0171 | 0.3539 ± 0.0630 | 0.9954 ± 0.0014 |

The strong baseline uses additive A similarity and centered N similarity in one
standard softmax, plus a learned null slot. It deliberately avoids cross-channel
dot products and has about 13% more parameters than Soft Chevron.

## Learned Chevron mechanism

| Measurement | Mean ± SD |
|---|---:|
| Target admission r | 0.9210 ± 0.0042 |
| Non-target admission r | 0.0055 ± 0.0006 |
| Null mass on matched examples | 0.7215 ± 0.0019 |
| Null mass on no-match examples | 0.9944 ± 0.0006 |
| Learned theta | 0.0944 ± 0.0020 |
| Learned sharpness k | 30.2188 ± 0.0084 |

The intended internal operation appeared in every seed. A attention retrieved
the plausible group. Normalized A/N mismatch admitted the matching retained
value and nearly eliminated all other values. When no template matched, almost
all mass flowed to V_null.

The relatively high null mass on matched examples is expected from the stated
non-renormalizing rule. A attention is shared across three plausible group
members; after the mismatch gate removes two members, their mass becomes null
rather than being redistributed. The learned value and answer projections can
decode the remaining admitted target contribution.

## Interpretation boundary

This is a promising initial result for the simplified Phase 7 equations. Soft
Chevron is slightly more accurate than the strong joint-attention baseline on
this constructed conditional-retrieval task while using fewer parameters. The
A-only result confirms that N matching is necessary.

It is not yet evidence of a general performance advantage. The task is designed
around separate retrieval and admission, and training includes explicit group
retrieval and gate targets. The joint baseline is also nearly perfect. Theta and
k remain close to their sensible initial values, so the experiment shows that
they are trainable but does not yet show that unconstrained optimization finds
the correct calibration from a poor starting point.

No online IDL write or persistent category creation occurs in this first Phase
7 experiment. It validates the learned forward path before retained-state
dynamics are added.

## Next tests

1. Remove the explicit gate loss and train from answer supervision alone.
2. Sweep template noise, category count, group size, and theta/k initialization.
3. Compare matched compute and training duration with joint attention.
4. Add persistence-controlled N writes and category allocation after the
   learned forward gate is robust.

