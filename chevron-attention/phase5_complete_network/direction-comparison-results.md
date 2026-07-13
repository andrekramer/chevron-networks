# Directional persistence comparison

The original robustness sweep exposed sensitivity to wrongly signed retention
signals. This follow-up compares three rules using the same three trained
models and five streams per model (15 runs per setting).

## Compared rules

**Hard reset** is the Phase 5 default and original comparison point. Any change
in proposed direction clears accumulated persistence.

**Two trace** maintains separate revoke and restore evidence:

```text
P+ <- beta P+ + (1-beta) positive_evidence
P- <- beta P- + (1-beta) negative_evidence
```

The trace corresponding to the present proposal controls its update gate. An
opposite observation decays prior evidence normally instead of erasing it.

**Signed hysteresis** stores one signed evidence average. Contradiction must
move that average through zero before retained change reverses direction.

## Clean comparison

| Rule | Preserve short | Consolidate revoke | Full revoke/restore | Revoke crossing | Restore crossing | N after revoke | N after restore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard reset | 100% | 100% | 100% | 29 steps | 36 steps | 0.0191 | 0.9675 |
| Two trace | 100% | 100% | 100% | 29 steps | 36 steps | 0.0191 | 0.9675 |
| Signed hysteresis | 100% | 100% | 100% | 29 steps | 54 steps | 0.0191 | 0.8696 |

Two trace exactly matches the clean hard-reset behavior. Signed hysteresis
adds an 18-step reversal delay because genuine restore evidence must first
cancel the retained revoke evidence.

## Random sign-flip corruption

| Flip probability | Rule | Preserve short | Consolidate revoke | Full revoke/restore | Worst answer accuracy | Revoke crossing | Restore crossing |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1% | Hard reset | 100% | 80% | 80% | 0.9167 | 37.7 | 50.7 |
| 1% | Two trace | 100% | 100% | 100% | 1.0000 | 29.8 | 37.0 |
| 1% | Signed hysteresis | 100% | 100% | 100% | 1.0000 | 29.8 | 54.6 |
| 5% | Hard reset | 100% | 40% | 20% | 0.9167 | 54.2 | 69.0 |
| 5% | Two trace | 100% | 100% | 100% | 1.0000 | 31.0 | 39.0 |
| 5% | Signed hysteresis | 100% | 100% | 100% | 1.0000 | 31.0 | 58.4 |
| 10% | Hard reset | 100% | 0% | 0% | 0.9125 | >70 | >70 |
| 10% | Two trace | 100% | 100% | 100% | 1.0000 | 35.0 | 41.6 |
| 10% | Signed hysteresis | 100% | 100% | 100% | 1.0000 | 35.2 | 63.2 |
| 20% | Hard reset | 100% | 0% | 0% | 0.9125 | >70 | >70 |
| 20% | Two trace | 100% | 100% | 100% | 1.0000 | 40.8 | 41.4 |
| 20% | Signed hysteresis | 100% | 100% | 20% | 0.9250 | 40.8 | >70 average |
| 30% | Hard reset | 100% | 0% | 0% | 0.9125 | >70 | >70 |
| 30% | Two trace | 100% | 100% | 100% | 1.0000 | 50.6 | 44.8 |
| 30% | Signed hysteresis | 100% | 80% | 0% | 0.9208 | 55.2 | >70 |

The two-trace rule completed every tested cycle through 30% independently
flipped observations. Its adaptation slowed smoothly as corruption increased,
rather than failing abruptly.

## Clean duration boundary

Hard reset and two trace were identical at every clean duration. Both
consolidated revoke and restore at 30 steps and above. Signed hysteresis
consolidated the revoke at 30 steps but required a 70-step restore to complete
the full reversal. At 30, 40, and 50 restore steps it remained below the active
decision boundary.

## Recommendation

Two directional persistence traces are the strongest rule in this experiment.
They remove the hard-reset failure without changing clean behavior or imposing
signed hysteresis's reversal delay. The current default remains `hard_reset`
so the original Phase 5 result is reproducible; `persistence_mode="two_trace"`
is the recommended candidate for the next phase and for tests with temporally
correlated or adversarial noise.

