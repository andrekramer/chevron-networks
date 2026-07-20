# Phase 8 locked confirmation protocol

## Frozen state

The architecture, trained-controller procedure, candidate mechanism, evidence
calibration, and all thresholds are frozen in `locked-config.json`. No result
from this run may change them.

Confirmation seeds are:

```text
1007, 1017, 1027, 1037, 1047, 1057, 1067, 1077, 1087, 1097,
1107, 1117, 1127, 1137, 1147, 1157, 1167, 1177, 1187, 1197
```

None appeared in Phase 6, Phase 7, Phase 8 development, parameter selection, or
the pre-lock robustness work.

## Primary conditions

Twenty paired seeds will run:

- blocked sustained novelty with two flipped N components;
- near-category novelty with one flipped N component.

Methods are Chevron immediate, Chevron quarantine, joint immediate, joint
quarantine, alpha quarantine, and the temporal oracle.

## Robustness conditions

The first ten confirmation seeds will additionally run:

- interleaved sustained novelty;
- noise 0.06, 0.08, and 0.10;
- three-flip category distance;
- transient duration 1, 3, 4, 5, and 6;
- retained capacity 8 rather than 9;
- doubled sustained duration of 16 observations per novel category;
- a two-slot candidate bank under three interleaved categories.

The four primary learned methods run in the robustness matrix.

## Statistical reporting

Report means and standard deviations, paired per-seed differences, win/tie/loss
counts, and deterministic paired bootstrap 95 percent confidence intervals.
Intervals contain 5,000 bootstrap resamples.

## Hypotheses

H1 passes if both quarantine methods have zero false consolidation at transient
durations one, three, and four.

H2 passes if both quarantine methods acquire at least 75 percent of blocked
novel categories and at least 60 percent of interleaved categories.

H3 passes if, in the near condition, Chevron quarantine has a positive paired
old-accuracy interval and a negative paired cross-write interval relative to
alpha quarantine.

H4 passes if Chevron quarantine has positive paired new-accuracy intervals
relative to both joint quarantine and immediate Chevron in the near condition.

H5 has no pass criterion. The default Chevron-versus-joint interval is reported
to prevent a near-category result from being generalized into universal
superiority.

