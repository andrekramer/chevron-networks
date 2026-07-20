# Phase 8 development result: provisional consolidation

## Status

This is development evidence, not a locked confirmation result. Seeds 107 and
117 selected the provisional parameters from an eight-point screen. Seed 127
was then added as a small held-aside development check. The reported three-seed
means therefore must not be treated as an unbiased final estimate.

The selected shared candidate parameters were:

```text
candidate capacity = 3
beta = 0.80
consolidation threshold = 0.25
minimum support = 5 observations
minimum eligible mass = 0.10
minimum distinct mismatch = 0.04
```

The sweep objective penalized false consolidation twice as strongly as the
other terms, then penalized sustained-learning failure, normalized acquisition
delay, old-category accuracy loss, and false splitting. It did not select by
final accuracy alone.

## Comparison

Each stream began with six retained categories. Isolated and transient streams
should not add the contradictory pattern. Sustained and near-category streams
contained three new categories that should be consolidated. Near categories
differed from an old category by one N component.

| Scenario | Method | Old accuracy | New accuracy | False consolidations | Novel categories retained | Delay |
|---|---|---:|---:|---:|---:|---:|
| Isolated | Chevron immediate | 1.000 | 0.333 | 1.00 | 1.00 | 1.0 |
| Isolated | Chevron quarantine | 1.000 | 0.000 | **0.00** | 0.00 | — |
| Isolated | Joint immediate | 1.000 | 0.222 | 0.67 | 0.67 | 1.0 |
| Isolated | Joint quarantine | 1.000 | 0.000 | **0.00** | 0.00 | — |
| Transient | Chevron immediate | 1.000 | 0.222 | 1.00 | 0.67 | 1.0 |
| Transient | Chevron quarantine | 1.000 | 0.000 | **0.00** | 0.00 | — |
| Transient | Joint immediate | 1.000 | 0.222 | 0.67 | 0.67 | 1.0 |
| Transient | Joint quarantine | 1.000 | 0.000 | **0.00** | 0.00 | — |
| Sustained | Chevron immediate | 1.000 | 0.556 | 0.00 | 1.67 | 1.0 |
| Sustained | Chevron quarantine | 1.000 | **0.889** | 0.00 | **2.67** | 5.0 |
| Sustained | Joint immediate | 1.000 | **0.889** | 0.00 | **2.67** | 1.0 |
| Sustained | Joint quarantine | 1.000 | 0.778 | 0.00 | 2.33 | 5.0 |
| Near | Chevron immediate | 1.000 | 0.222 | 0.00 | 0.67 | 1.0 |
| Near | Chevron quarantine | 1.000 | **0.778** | 0.00 | **2.33** | 5.0 |
| Near | Joint immediate | 1.000 | 0.000 | 0.00 | 0.00 | — |
| Near | Joint quarantine | 1.000 | 0.000 | 0.00 | 0.00 | — |

The temporal oracle retained all three sustained and near categories after
exactly five observations and reached 1.0 old and new accuracy.

## What the development result suggests

Provisional persistence solved the intended temporal distinction on these
seeds. Both quarantine systems rejected every isolated and three-observation
disturbance, while both still learned ordinary sustained novelty after a
five-observation delay.

Chevron quarantine showed the most interesting change in the difficult near
condition. It retained seven of the nine possible new categories across the
three seeds, versus two for Chevron immediate and zero for either joint method.
The result appeared on all three seeds: Chevron quarantine retained two, three,
and two near categories respectively.

This is not yet evidence that quarantine generally belongs specifically to
Chevron. On ordinary sustained novelty, Chevron quarantine and joint immediate
both reached 0.889 new accuracy, while joint quarantine reached 0.778. The
specific candidate is the combination of temporal persistence with
slot-specific A/N assent under close category interference.

## Assent ablation

Alpha quarantine received the same candidate logic but wrote alpha mass to
retained slots without multiplying by assent `r`. Its old-category accuracy
fell to 0.637 after an isolated contradiction, 0.717 after a transient, 0.372
under sustained novelty, and 0.265 in the near condition. Mean cross-category
write mass was roughly 0.72–0.79, compared with 0.04–0.07 for Chevron
quarantine.

This supports the Phase 7 mechanistic result: temporal quarantine controls
when a category is added, while slot-specific assent is still needed to
protect existing templates during ordinary writes.

## Required checks before confirmation

The parameters should not yet be locked. The next development pass must test:

1. interleaved rather than blocked sustained categories;
2. transient durations around the five-observation boundary;
3. observation noise at 0.06, 0.08, and 0.10;
4. category distance of one, two, and three flipped N components;
5. whether threshold-centering gives the joint controller a stable candidate
   evidence scale across seeds;
6. candidate-bank capacity and replacement pressure.

If one shared configuration remains on the stability-plasticity frontier, it
can then be frozen for new-seed confirmation. No claim beyond a promising
development result is justified before that run.

Raw per-stream results and the full selected configuration are stored in
`development-results.json`; all screened settings and objective components are
stored in `development-sweep.json`.

