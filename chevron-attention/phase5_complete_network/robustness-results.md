# Phase 5 robustness results

The robustness suite was run with three independently trained models (seeds
`1`, `7`, and `13`) and five independent online streams per model (seeds `101`,
`211`, `307`, `401`, and `503`). Thus each online setting contains 15 runs.
Each model trained for 700 steps.

An earlier default-condition audit used ten trained models and five streams per
model. All 50 integrated-IDL runs achieved perfect short preservation, long
revocation consolidation, and restoration. The sweeps below deliberately move
away from that saturated default.

## Generalization beyond the training shape

Training used six facts and at most four controls. The frozen models were
evaluated on fresh examples with different memory and control-set sizes.

| Facts | Maximum controls | Mean answer accuracy | Worst answer | Worst retrieval | Worst context |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 10 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The learned address-and-control mechanism generalizes across the full available
key set, although it does not test unseen key identities beyond the trained
embedding vocabulary.

## Duration boundary

| Revoke duration | Integrated retained N | Integrated preservation | Always-update N | Always-update preservation |
|---:|---:|---:|---:|---:|
| 2 | 0.9999 | 100% | 0.8464 | 100% |
| 5 | 0.9997 | 100% | 0.6591 | 100% |
| 10 | 0.9982 | 100% | 0.4344 | 0% |
| 15 | 0.9913 | 100% | 0.2863 | 0% |
| 20 | 0.9647 | 100% | 0.1887 | 0% |
| 30 | 0.7313 | 100% | 0.0820 | 0% |
| 40 | 0.3746 | 0% | 0.0356 | 0% |

| Sustained duration | Integrated retained N | Integrated consolidation | Fixed-slow N | Fixed-slow consolidation |
|---:|---:|---:|---:|---:|
| 20 | 0.8243 | 0% | 0.7859 | 0% |
| 30 | 0.4608 | 100% | 0.7252 | 0% |
| 40 | 0.2097 | 100% | 0.6692 | 0% |
| 50 | 0.0924 | 100% | 0.6176 | 0% |
| 60 | 0.0411 | 100% | 0.5699 | 0% |
| 70 | 0.0191 | 100% | 0.5259 | 0% |
| 90 | 0.0066 | 100% | 0.4479 | 100% |
| 120 | 0.0046 | 100% | 0.3520 | 100% |

This locates the default IDL transition between roughly 20 and 40 repeated
steps. A 40-step event is no longer temporary under these parameters; that is
the mechanism's temporal decision boundary, not a claim that “short” has a
universal duration.

## Imperfect retention signals

Noise is injected only into the contextual signal passed to retained-state
updating. The neural network still controls immediate contextual behavior, so
probe failures isolate consolidation rather than retrieval.

| Perturbation | Level | Worst overall accuracy | Preserve short | Consolidate revoke | Full revoke/restore cycle |
|---|---:|---:|---:|---:|---:|
| Gaussian | 0.05 | 1.0000 | 100% | 100% | 100% |
| Gaussian | 0.10 | 1.0000 | 100% | 100% | 100% |
| Gaussian | 0.20 | 1.0000 | 100% | 100% | 100% |
| Gaussian | 0.35 | 0.9250 | 100% | 80% | 46.7% |
| Dropout | 10% | 1.0000 | 100% | 100% | 100% |
| Dropout | 25% | 1.0000 | 100% | 100% | 100% |
| Dropout | 50% | 0.9125 | 100% | 6.7% | 6.7% |
| Sign flip | 1% | 0.9167 | 100% | 80% | 80% |
| Sign flip | 5% | 0.9167 | 100% | 40% | 20% |
| Sign flip | 10% | 0.9125 | 100% | 0% | 0% |
| Sign flip | 20% | 0.9125 | 100% | 0% | 0% |

The sign-flip result is a useful negative finding. The original IDL update
resets directional persistence when the proposed direction reverses. That
protects against stale evidence during a real revoke-to-restore transition but
makes consolidation sensitive to isolated wrongly signed observations. The
follow-up in `direction-comparison-results.md` tests two alternatives.

## IDL parameter sensitivity

At `eta_n=0.08`, every tested `(beta, threshold)` pair succeeded except the
slowest persistence setting, `beta=0.995`, combined with thresholds of `0.35`
or above. Those settings never accumulated enough evidence during the 70-step
sustained episode.

| beta | θ=0.20 | θ=0.30 | θ=0.35 | θ=0.40 | θ=0.50 |
|---:|---:|---:|---:|---:|---:|
| 0.970 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| 0.980 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| 0.985 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| 0.990 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| 0.995 | 100/100 | 100/100 | 100/0 | 100/0 | 100/0 |

Cells show short-preservation percentage / long-consolidation percentage.

The slow-update boundary is also visible. “Full cycle” requires successful
revocation before final active behavior is credited as restoration:

| eta_n | Preserve short | Consolidate revoke | Full cycle | N after revoke | N after restore |
|---:|---:|---:|---:|---:|---:|
| 0.002 | 100% | 0% | 0% | 0.9069 | 0.9072 |
| 0.005 | 100% | 0% | 0% | 0.7831 | 0.8181 |
| 0.010 | 100% | 0% | 0% | 0.6125 | 0.7447 |
| 0.020 | 100% | 100% | 100% | 0.3735 | 0.7291 |
| 0.040 | 100% | 100% | 100% | 0.1370 | 0.8411 |
| 0.080 | 100% | 100% | 100% | 0.0191 | 0.9675 |
| 0.120 | 100% | 100% | 100% | 0.0053 | 0.9917 |
| 0.160 | 100% | 100% | 100% | 0.0029 | 0.9961 |

## Conclusion

The positive Phase 5 result is not seed-specific and survives substantial
changes in memory size, control-set size, duration, moderate continuous noise,
and moderate missing retention signals. The experiment also identifies clear
limits: temporal evidence has a tunable boundary, very slow persistence/update
settings fail to consolidate within the tested horizon, and hard directional
reset is brittle to wrongly signed observations. These boundaries are more
informative than another saturated perfect-accuracy seed table.
