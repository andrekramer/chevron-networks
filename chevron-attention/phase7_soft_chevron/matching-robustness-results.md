# Phase 7.3 results: learning the A/N comparison space

Phase 7.2 still gave Soft Chevron a favorable starting geometry: the A and N
matching projections began as the same centered identity transform. This
experiment removes that assumption and then perturbs the representations.

Every condition uses answer labels only, 700 training steps, the same five
seeds, and fresh held-out episodic memories. `Target r` is admission of the
correct retained template; `decoy r` is mean admission of other templates.

## Random matching projections

| A/N projection initialization | Answer accuracy | Target mismatch | Decoy mismatch | Target r | Decoy r | No-match null mass |
|---|---:|---:|---:|---:|---:|---:|
| Centered identity | 1.0000 ± 0.0000 | 0.0147 ± 0.0016 | 0.2582 ± 0.0016 | 0.9340 ± 0.0035 | 0.0205 ± 0.0009 | 0.9798 ± 0.0008 |
| Shared random | 1.0000 ± 0.0000 | 0.0125 ± 0.0079 | 0.1521 ± 0.0044 | 0.8425 ± 0.0250 | 0.0793 ± 0.0089 | 0.9211 ± 0.0086 |
| Independent random | 1.0000 ± 0.0000 | 0.0315 ± 0.0018 | 0.1573 ± 0.0025 | 0.8130 ± 0.0081 | 0.0916 ± 0.0031 | 0.9082 ± 0.0031 |

The model does not need to be handed a common A/N coordinate system. Even when
the two matching projections begin independently, answer-only learning finds a
comparison geometry that separates matching and non-matching templates and
solves every held-out query in all five runs.

The identity initialization still produces the cleanest internal gate. Its
target and decoy admissions are farther apart and it routes more unmatched mass
to the explicit null value. Random initialization therefore establishes
learnability, not equivalence of the learned mechanisms.

## Training and testing with representation noise

Both current A evidence and retained N templates receive Gaussian noise with
standard deviation 0.05 during training and evaluation.

| Initialization | Answer accuracy | Matched accuracy | No-match accuracy | Target r | Decoy r | No-match null mass |
|---|---:|---:|---:|---:|---:|---:|
| Centered identity | 0.9995 ± 0.0003 | 0.9995 ± 0.0004 | 0.9994 ± 0.0008 | 0.8960 ± 0.0028 | 0.0354 ± 0.0013 | 0.9651 ± 0.0010 |
| Independent random | 0.9998 ± 0.0002 | 1.0000 ± 0.0000 | 0.9991 ± 0.0009 | 0.7776 ± 0.0112 | 0.0887 ± 0.0024 | 0.9111 ± 0.0022 |

Both variants remain effectively perfect when noise is present during
learning. The result is stable across seeds and is not an artifact of one
favorable initialization.

## Unseen representation noise

For the harder distribution-shift condition, models train on the original
clean current representation and template noise of 0.015. At evaluation time,
both channels receive previously unseen noise with standard deviation 0.08.

| Initialization | Answer accuracy | Matched accuracy | No-match accuracy | Target mismatch | Decoy mismatch | Target r |
|---|---:|---:|---:|---:|---:|---:|
| Centered identity | 0.9158 ± 0.0039 | 0.8877 ± 0.0054 | 1.0000 ± 0.0000 | 0.0755 ± 0.0012 | 0.2780 ± 0.0015 | 0.6818 ± 0.0054 |
| Independent random | 0.9887 ± 0.0029 | 0.9879 ± 0.0033 | 0.9908 ± 0.0052 | 0.0563 ± 0.0017 | 0.1568 ± 0.0023 | 0.6678 ± 0.0118 |

The independently initialized matcher generalizes substantially better in this
shift: 98.87% versus 91.58% overall accuracy. It learns a less sharply separated
gate on clean data, but maps noisy matching targets to a lower normalized
mismatch and preserves enough admitted value mass to answer them. This is
evidence that learning the comparison space can improve robustness rather than
merely recover the supplied identity metric.

## Strongest supported claim

> On this synthetic episodic category task, answer-only Soft Chevron Attention
> reliably learns both retrieval and an explicit A/N admission mechanism from
> independently initialized matching projections. It averages 98.87% accuracy
> across five seeds under the tested unseen representation shift and outperforms
> its identity-initialized form in that condition.

This is not yet a general advantage over standard attention. The earlier joint
attention baseline reaches the same clean answer accuracy, and it has not yet
been subjected to this representation-shift comparison. The task also has
fixed synthetic category structure and no online memory writes. The next
litmus test should therefore compare Soft Chevron and joint attention under the
same noise shifts before adding online IDL category formation.
