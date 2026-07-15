# Phase 6.1 results: recurring category drift

The initial category-allocation task established that persistent creation beats
immediate writes, but persistent single-template attention tied Chevron. This
follow-up asks for something a single template cannot trivially provide: track
a category's current surface form, preserve its retained form through a brief
shift, and consolidate the same shift when it persists.

Five seeded streams (`7`, `17`, `27`, `37`, and `47`) used the same category
geometry with independent observation noise. Category zero underwent a
ten-observation shift, a return to its base form, and then a seventy-observation
shift. Predictions preceded supervised online updates. All probes were
read-only.

## Result

| Method | Short shift online | Current form after short shift | Base form recovered | Minimum of current/base | Long shift online | Long adaptation steps | Final current form | Retained N after long shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard-search Chevron A/N | 0.6000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9400 ± 0.0064 | 4.2 ± 0.4 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Soft-contrast Chevron A/N | 0.9000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9857 ± 0.0000 | 1.0 ± 0.0 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Standard dual-trace attention | 0.9000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9857 ± 0.0000 | 1.0 ± 0.0 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Persistent single template | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8429 ± 0.0101 | 11.0 ± 0.7 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Fast single template | 0.9000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9857 ± 0.0000 | 1.0 ± 0.0 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Slow single template | 0.1000 ± 0.0707 | 0.9600 ± 0.0548 | 1.0000 ± 0.0000 | 0.9600 ± 0.0548 | 0.9600 ± 0.0120 | 2.8 ± 0.8 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

`Retained N after long shift` forces dual-trace models to answer using N alone;
it therefore verifies consolidation rather than success through the fast A
trace.

## Interpretation

Two traces cleanly span the tested tradeoff. A follows the temporary surface
form, N recovers the original form, and persistent contrast later moves N to
the sustained form. Fast single-template attention follows the shift but loses
the original. Persistence-gated single-template attention preserves the
original but cannot express the temporary form until its only template moves.

However, ordinary softmax attention over duplicated A and N memory tokens ties
the soft-contrast Chevron on every metric and slightly outperforms the hard
vigilance/search Chevron during online adaptation. The hard reset needs 4.2
observations to settle; both soft methods need one.

This yields a stronger but still bounded conclusion:

> In this controlled drift task, separate fast and retained category traces
> solve a stability/plasticity tradeoff that one template does not solve as
> cleanly. Normalized contrast can gate retention without hurting performance
> when used softly. Hard ART-style reset/search is unnecessary here, and the
> same two traces work equally well as ordinary dual-memory attention.

The result supports dual timescales and soft contrast gating. It does not yet
support a performance advantage for Chevron routing over a well-constructed
standard-attention baseline.

## Next decisive test

The next task should make the *interaction* between selection and retained
matching consequential: several A keys should be plausible, only one N
template should satisfy the top-down constraint, and the correct alternative
should change with context. That would test whether ordered veto/search adds
anything that soft attention over A/N tokens cannot learn.

