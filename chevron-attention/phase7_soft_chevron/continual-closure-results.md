# Phase 7.6 results: closure tests for persistent writes

Phase 7.5 found a 12.6-point final-accuracy advantage for Chevron over joint
attention using either its native null mass or downstream P(IDK) as a write
veto. Three questions remained:

1. Can a standard-attention controller trained specifically for writes close
   that gap?
2. Does Chevron's slot-specific `alpha * r` admission cause the safer writes?
3. Does the result survive changes in noise, category distance, capacity, and
   stream order?

These closure experiments answer all three. They also narrow the Phase 7 claim.

## A dedicated, parameter-matched write controller

The new joint baseline has a learned veto head trained explicitly to predict
merge versus allocate. It receives nine permutation-invariant statistics of
the joint A/N attention field, including maxima, margin, dispersion, entropy,
and active-memory size. Training randomizes the number of active slots from six
to nine and guarantees that a matching target remains visible.

The controller has 177 parameters. Joint attention uses width 35, producing
3,726 total parameters versus Chevron's 3,759. The controller receives direct
no-match supervision, which is favorable compared with Chevron's answer-only
gate learning.

### Ten-seed default comparison

| Measurement | Soft Chevron | Learned joint controller |
|---|---:|---:|
| Final clean accuracy | 0.8775 ± 0.0630 | 0.8439 ± 0.1313 |
| Final shifted accuracy | 0.8689 ± 0.0630 | 0.8411 ± 0.1356 |
| False-merge rate | **0.0355 ± 0.0683** | 0.3712 ± 0.2768 |
| False-split rate | **0.0384 ± 0.0309** | 0.1255 ± 0.1080 |
| Wrong-category write mass | **0.0667 ± 0.0154** | 0.3423 ± 0.0305 |
| Categories retained | 7.90 ± 0.57 | 7.60 ± 1.17 |
| Template MSE | 0.0047 ± 0.0007 | 0.0060 ± 0.0041 |

Chevron's paired clean-accuracy difference is only `+0.0336 ± 0.1582`. It wins
four seeds, loses four, and ties two. Final accuracy is therefore unresolved:
the dedicated controller closes the earlier performance gap.

Chevron still produces a substantially cleaner write trajectory. It sends
6.7% of write mass to wrong-category slots versus 34.2% for the controller and
has lower merge and split rates. In this short, capacity-limited task, those
mechanistic advantages do not produce a reliable final-accuracy difference
against a controller trained specifically for writing.

## Causal write ablations

All Chevron variants use the same trained network and stream.

- `alpha_only` retains the learned novelty/allocation score but removes `r`
  from per-slot writes.
- `fixed_gate` replaces learned projected matching with raw normalized mismatch
  at fixed `theta=0.10`, `k=30`.
- `misaligned_gate` rotates learned admission values onto the wrong active
  slots.
- `oracle_write` uses the revealed label to allocate and update only the
  correct category, providing an upper bound.

| Write policy | Final clean | Wrong write mass | Template MSE | Categories retained |
|---|---:|---:|---:|---:|
| Learned `alpha * r` | **0.8775 ± 0.0630** | **0.0667 ± 0.0154** | **0.0047 ± 0.0007** | **7.90 ± 0.57** |
| Alpha only | 0.8100 ± 0.0920 | 0.7852 ± 0.0231 | 0.0090 ± 0.0042 | 7.30 ± 0.82 |
| Fixed raw gate | 0.6683 ± 0.1029 | 0.0186 ± 0.0101 | 0.0057 ± 0.0010 | 6.10 ± 0.88 |
| Misaligned learned gate | 0.7233 ± 0.1172 | 0.1356 ± 0.0163 | 0.0063 ± 0.0020 | 6.60 ± 0.84 |
| Oracle write | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0002 ± 0.0000 | 9.00 ± 0.00 |

Removing `r` from the writes increases wrong-category mass by almost twelve
times while leaving the same learned novelty calibration in place. The final
accuracy difference is `+0.0675 ± 0.1198`; learned admission wins four seeds,
ties five, and loses one. Thus the contamination effect is much clearer than
the discrete final-accuracy effect.

Learned admission beats the fixed raw gate by 20.9 points, winning eight seeds
and tying two. It beats the slot-misaligned gate by 15.4 points, winning eight,
tying one, and losing one. Both results support learned, slot-specific A/N
assent rather than a generic global novelty threshold.

The oracle reaches 100%, leaving room for better learned allocation and showing
that remaining errors are caused by write decisions rather than read capacity.

## Five-seed robustness matrix

The matrix reuses the first five trained seeds and changes one stream property
at a time.

| Condition | Soft Chevron | Joint controller | Chevron delta | Chevron wins |
|---|---:|---:|---:|---:|
| Default | 0.8667 ± 0.0930 | 0.8444 ± 0.1267 | +0.0222 | 1/5 |
| Shift noise 0.06 | 0.9111 ± 0.0930 | 0.8722 ± 0.1708 | +0.0389 | 2/5 |
| Shift noise 0.10 | 0.7772 ± 0.0786 | 0.8150 ± 0.1046 | -0.0378 | 1/5 |
| Near categories, one flipped bit | **0.8950 ± 0.1451** | 0.6683 ± 0.0940 | **+0.2267** | **5/5** |
| Far categories, three flipped bits | 0.8222 ± 0.0994 | 0.8611 ± 0.1547 | -0.0389 | 1/5 |
| Eight-slot capacity | 0.8000 ± 0.0930 | 0.8006 ± 0.1437 | -0.0006 | 2/5 |
| Blocked novel categories | 0.9111 ± 0.0930 | 0.7778 ± 0.1111 | +0.1333 | 3/5 |
| Doubled shifted stream | 0.8889 ± 0.1111 | 0.8222 ± 0.0994 | +0.0667 | 3/5 |

There is no general robustness win. The controller is slightly better under
the highest noise and when categories are farther apart; capacity pressure is
a tie. Chevron's clearest advantage occurs when new categories are very close
to retained categories, where stability and plasticity are in direct tension.
It wins all five near-category seeds by 22.7 points on average.

## Revised conclusion

The Phase 7.5 claim that Chevron generally improves continual accuracy over
standard attention does not survive the dedicated-controller test.

The strongest supported claim is narrower:

> Learned, slot-specific A/N admission strongly reduces cross-category memory
> writes and is especially useful when novel categories closely resemble
> retained ones. It outperforms alpha-only, fixed, and misaligned write rules,
> but does not outperform a parameter-matched learned joint-attention write
> controller across the default task or all robustness conditions.

This still supports Chevron Attention as a useful stability-plasticity
inductive bias. It does not establish general architectural superiority. The
next step, if pursued, should move to less synthetic data rather than further
tuning this benchmark.
