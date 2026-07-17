# Phase 7.5 results: when attention writes become consequential

> **Closure note:** Phase 7.6 adds a parameter-matched joint-attention
> controller trained explicitly for write veto. That controller closes the
> default final-accuracy gap. The Phase 7.5 result remains valid against the two
> baselines tested here, but its general accuracy claim is superseded by
> `continual-closure-results.md`.

Phase 7.4 found no final-answer advantage for Soft Chevron when memories were
read-only. Both Chevron and joint attention remained near 99% accurate under
unseen representation noise. This experiment makes the admission decision
consequential by allowing it to change persistent memory.

## Experiment

Each model first receives the same 700 steps of answer-only episodic training
used in the earlier Phase 7 comparisons. Soft Chevron starts with independent
A/N matching projections. The learned models are then frozen and placed in a
continual category stream with a nine-slot memory:

1. six related categories begin as stable retained memories;
2. those categories receive 180 online adaptation observations;
3. three new categories sharing the same A address groups arrive during 180
   mixed observations;
4. all nine categories continue for 270 observations under unseen noise 0.08;
5. clean and shifted final probes run without further writes.

Category allocation is label-free. The observed label is supplied only after
the decision and becomes the stored V value when a new slot is allocated. A
held-out episodic calibration set chooses each policy's novelty threshold.

For Chevron, retained writes use its learned admitted mass:

```text
w_j = alpha_j * r_j
N_j <- (1 - eta_N * w_j) N_j + eta_N * w_j * x_N
```

The A key uses the same rule with its own write rate. Remaining mass determines
whether to allocate a new category.

Two joint-attention controls receive the same stream, capacity, values, and
write rates:

- `joint_null_write` uses standard attention mass and its learned null slot;
- `joint_output_write` is the stronger control. It may convert downstream
  P(IDK) into a write veto and rescales total write mass by `1 - P(IDK)`.

The stronger control tests whether the answer head can turn the compensation
seen in Phase 7.4 into a safe memory policy.

## Ten-seed results

| Measurement | Soft Chevron | Joint null write | Joint output veto |
|---|---:|---:|---:|
| Base online accuracy | 1.0000 ± 0.0000 | 0.9561 ± 0.0287 | 0.9872 ± 0.0404 |
| Novel-phase online accuracy | 0.9478 ± 0.0568 | 0.7289 ± 0.0333 | 0.7983 ± 0.0945 |
| Shift-phase online accuracy | 0.8952 ± 0.0816 | 0.7263 ± 0.0270 | 0.7519 ± 0.1140 |
| Final old-category accuracy | 0.8496 ± 0.1224 | 0.7117 ± 0.1333 | 0.8137 ± 0.1624 |
| Final new-category accuracy | 0.9000 ± 0.1610 | 0.6975 ± 0.1849 | 0.5933 ± 0.3052 |
| Final clean accuracy | **0.8664 ± 0.0702** | 0.7069 ± 0.0910 | 0.7403 ± 0.0792 |
| Final shifted accuracy | **0.8597 ± 0.0703** | 0.7000 ± 0.0872 | 0.7317 ± 0.0823 |
| Categories retained out of 9 | **7.80 ± 0.63** | 6.40 ± 0.84 | 6.70 ± 0.67 |

Against the stronger output-veto baseline, Chevron gains 12.61 ± 6.66
percentage points on the paired clean final probes and wins nine of ten seeds,
with one tie. On shifted probes it gains 12.81 ± 7.61 points and wins nine of
ten seeds.

## Why the memories diverge

| Write failure | Soft Chevron | Joint null write | Joint output veto |
|---|---:|---:|---:|
| False-merge rate | **0.0510 ± 0.1471** | 0.0429 ± 0.1355 | 0.8346 ± 0.2132 |
| False-split rate | 0.0284 ± 0.0326 | 0.2696 ± 0.0763 | **0.0236 ± 0.0173** |
| Write mass sent to wrong categories | **0.0673 ± 0.0154** | 0.3839 ± 0.0187 | 0.4310 ± 0.0590 |
| Template MSE after the stream | **0.0046 ± 0.0007** | 0.0054 ± 0.0008 | 0.0157 ± 0.0099 |
| Evictions | 45.1 ± 44.6 | 253.0 ± 61.9 | **18.5 ± 15.2** |

The native joint null signal recognizes novel cases but is poorly calibrated as
the active memory changes. It repeatedly splits familiar categories, fills the
memory with duplicates, and evicts retained categories.

The downstream-IDK policy avoids most false splits, but often suppresses a new
slot while its attention still writes into existing categories. It falsely
merges new-category encounters in 83% of opportunities and sends 43% of its
write mass to slots carrying a different category value.

Chevron balances the two errors more effectively. Its normalized A/N admission
sends only 6.7% of write mass to wrong categories, about one sixth of the
stronger joint baseline, while keeping both merge and split rates relatively
low.

## Perfect-memory control

| Read-only oracle memory | Soft Chevron | Joint attention |
|---|---:|---:|
| Clean accuracy | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Shifted accuracy | 0.9972 ± 0.0060 | 1.0000 ± 0.0000 |

Both trained architectures can read an uncontaminated final memory. The
continual-learning gap therefore arises from the online write and allocation
trajectory, not an intrinsic inability of joint attention to classify these
nine categories.

## Strongest supported claim

> In this controlled continual category task, learned Soft Chevron admission
> makes persistence-controlled writes materially safer than standard joint
> attention, including a stronger baseline that converts downstream IDK
> confidence into a write veto. Across ten seeds, Chevron retains more
> categories and improves final accuracy by about 12.6 percentage points.

This is the first Phase 7 result supporting a functional stability-plasticity
advantage rather than only an architectural distinction.

## Boundary

The result remains task-specific. Category addresses and templates are
synthetic, the six starting memories are supplied, new V labels are revealed
after observation, and novelty thresholds are calibrated on held-out episodic
data. The frozen models are not trained end-to-end through their write
trajectories, and the comparison does not yet include replay-equipped MLPs or
larger Transformers. These are follow-on tests, not claims already established
by this experiment.
