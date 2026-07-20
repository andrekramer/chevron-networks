# Phase 8 pre-lock robustness result

> This pre-lock failure led to the focused revision recorded in
> `revision-results.md`. The revised configuration subsequently passed the
> development-readiness checks and was frozen in `locked-config.json`.

## Decision

The current Phase 8 system is not ready for locked confirmation.

Chevron quarantine passed the predeclared stability, blocked acquisition,
interleaved acquisition, old-memory, and near-category checks. The shared
configuration failed because joint quarantine under-acquired ordinary novelty,
especially when three candidate categories were interleaved.

This was a five-seed development run using seeds 107, 117, 127, 137, and 147.
It is not confirmation evidence.

## Stability-plasticity boundary

The temporal rule behaved cleanly around its declared minimum support:

| Disturbance duration | Chevron immediate false consolidation | Chevron quarantine | Joint immediate | Joint quarantine |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | **0.00** | 0.60 | **0.00** |
| 3 | 1.00 | **0.00** | 0.60 | **0.00** |
| 4 | 1.00 | **0.00** | 0.60 | **0.00** |
| 5 | 1.00 | 0.80 | 0.60 | 0.40 |
| 6 | 1.00 | 0.80 | 0.60 | 0.40 |

Neither quarantine system consolidated evidence lasting fewer than five
observations. Consolidation began at observation five, exactly where the
minimum-support rule placed the operational boundary. This is not a claim that
five is universally correct; it demonstrates that the architecture can impose
a transparent temporal boundary.

## Sustained and interleaved novelty

| Condition | Method | Old accuracy | New accuracy | Categories retained |
|---|---|---:|---:|---:|
| Blocked | Chevron immediate | 1.000 | 0.667 | 2.00 / 3 |
| Blocked | Chevron quarantine | **1.000** | **0.933** | **2.80 / 3** |
| Blocked | Joint immediate | 1.000 | 0.867 | 2.60 / 3 |
| Blocked | Joint quarantine | 1.000 | 0.733 | 2.20 / 3 |
| Interleaved | Chevron immediate | 1.000 | 0.667 | 2.00 / 3 |
| Interleaved | Chevron quarantine | **1.000** | **0.867** | **2.60 / 3** |
| Interleaved | Joint immediate | 1.000 | 0.867 | 2.60 / 3 |
| Interleaved | Joint quarantine | 1.000 | 0.333 | 1.00 / 3 |

Chevron quarantine improved on immediate Chevron in both orders while
preserving every old category. Joint quarantine imposed stability but lost too
much plasticity under interleaving.

The joint controller's centered evidence still separated the streams: mean
interleaved event evidence was 0.440 versus 0.053 on familiar observations.
The failure is therefore not simply absence of a novelty signal. The evidence
is weaker than Chevron's and decays between interleaved observations before it
can reliably cross the shared persistence threshold.

## Category distance

| Flipped N components | Chevron immediate new accuracy | Chevron quarantine | Joint immediate | Joint quarantine |
|---:|---:|---:|---:|---:|
| 1 | 0.467 | **0.867** | 0.000 | 0.000 |
| 2 | 0.667 | **0.933** | 0.867 | 0.733 |
| 3 | 0.667 | 0.867 | **1.000** | 0.867 |

The Phase 7 near-category pattern replicated on all five development seeds.
Chevron quarantine retained 2, 3, 2, 3, and 3 of the three possible close
categories. Both joint variants retained none. As categories became easier and
farther apart, joint attention caught up or became better.

This supports a specific inductive-bias interpretation, not general Chevron
superiority.

## Noise

Chevron quarantine retained 2.8, 2.2, and 1.8 of three novel categories at
noise 0.06, 0.08, and 0.10 respectively. New-category accuracy was 0.933,
0.733, and 0.598. Immediate Chevron fell more sharply to 0.531, 0.267, and
0.133. Chevron quarantine maintained 1.000 old-category accuracy at every
noise level.

Joint quarantine retained 2.0, 1.6, and 1.6 categories at those noise levels.
It also maintained old accuracy, but was generally less plastic than joint
immediate.

## Candidate capacity

Three interleaved novel categories required a three-slot provisional bank.
With one candidate slot neither quarantine method learned a category. With two
slots, Chevron learned 0.2 and joint learned 0.8 categories on average. With
three slots, those means rose to 2.6 and 1.0.

This is expected state pressure rather than a free capacity result: a bank must
be able to preserve the concurrent candidates it is asked to distinguish.

## Diagnostic threshold test

The original shared consolidation threshold was 0.25. A single diagnostic run
lowered it to 0.20 while leaving minimum support at five, so no category could
consolidate earlier.

That change:

- raised joint blocked acquisition from 2.2 to 2.4 categories;
- raised joint interleaved acquisition from 1.0 to 1.2 categories;
- preserved zero false consolidation at durations one through four;
- did not reach the predeclared 60 percent interleaved acquisition criterion.

The diagnostic therefore did not make the shared configuration lock-ready.
Further threshold tuning on these seeds would risk fitting the development
matrix rather than testing the mechanism.

## Additional issue: provisional churn

Chevron's event/familiar eligible-mass separation was 0.616 versus 0.197 in the
interleaved condition, but familiar observations still entered provisional
state frequently. The bank averaged 38.6 candidate replacements per stream.
Distinctness checks prevented these candidates from changing retained memory,
but the churn is unnecessary and should be resolved before confirmation.

A candidate that accumulates enough support but matches an existing retained
template should probably be rejected or merged and cleared, rather than held
until it displaces another candidate. This is a state-management correction,
not evidence for changing the retained write rule.

## Next decision

One development revision is warranted before parameters are frozen:

1. clear or merge mature candidates that are not distinct from retained
   memory, eliminating Chevron's familiar-candidate churn;
2. specify a fair method-specific evidence calibration for the joint
   controller, as permitted by the Phase 8 plan when signal scales are not
   shared, or explicitly treat joint quarantine's reduced plasticity as the
   baseline result;
3. rerun only the affected churn, transient-boundary, blocked, and interleaved
   checks on development seeds;
4. if those checks pass without more tuning, freeze everything and move to new
   confirmation seeds.

The raw primary matrix is in `robustness-results.json`. The one-threshold
diagnostic is in `robustness-results-tau20.json`.
