# Phase 7: learned soft Chevron Attention

Phase 7 begins with the simplified differentiable rules selected by Phase 6:

```text
alpha = softmax(Q_A @ K_A^T)
r = sigmoid(k * (theta - M(A,N)))
y = sum(alpha * r * V_N) + remaining_mass * V_null
```

The first learned task contains fresh episodic category memories. A addresses
identify a plausible group containing several members. Current match evidence
must then agree with a retained N template before that member's value can
participate. Some queries deliberately match no template and must use the null
value.

All visible Phase 7 computation is differentiable and trained jointly:

- learned Q_A and K_A projections;
- learned A and N matching projections;
- learned normalized-mismatch threshold theta and sharpness k;
- learned V_N values, null value, and answer head.

The primary baseline is standard joint A/N attention with a learned null slot.
An A-only attention model measures whether the N matching signal is actually
necessary.

Run the tests:

```bash
.venv/bin/python -m unittest phase7_soft_chevron.test_experiment -q
```

Run five seeds:

```bash
.venv/bin/python -m phase7_soft_chevron.experiment --device auto
```

Methods may be run separately with `--methods soft_chevron`,
`--methods joint_attention`, or `--methods a_only_attention`.

The first five-seed run is recorded in `initial-results.md`.

Compare full auxiliary supervision, retrieval-only supervision, and pure answer
supervision:

```bash
.venv/bin/python -m phase7_soft_chevron.supervision_comparison
```

The executed five-seed ablation is recorded in `supervision-results.md`.

Sweep answer-only learning across calibrated, overly open, overly closed, soft,
sharp, and saturated gate initializations:

```bash
.venv/bin/python -m phase7_soft_chevron.initialization_comparison
```

The executed five-seed initialization sweep is recorded in
`initialization-results.md`.

Test favorable identity, shared-random, and independent-random A/N matching
projections under clean, trained-noise, and out-of-distribution noise:

```bash
.venv/bin/python -m phase7_soft_chevron.matching_robustness
```

The executed five-seed comparison is recorded in
`matching-robustness-results.md`.

Compare independently initialized Soft Chevron directly with the strong joint
A/N attention baseline under the same representation shifts:

```bash
.venv/bin/python -m phase7_soft_chevron.attention_shift_comparison
```

The executed five-seed litmus test is recorded in
`attention-shift-results.md`.

Use learned admission to control persistent category writes, and compare it
with both native-null and downstream-IDK joint-attention write policies:

```bash
.venv/bin/python -m phase7_soft_chevron.continual_memory
```

The executed ten-seed continual-learning comparison is recorded in
`continual-memory-results.md`.

Run the Phase 7 closure suite with a parameter-matched learned write
controller, causal write ablations, and a robustness matrix:

```bash
.venv/bin/python -m phase7_soft_chevron.continual_closure
```

The closure result and revised claim are recorded in
`continual-closure-results.md`.

The plain-text Phase 7 and 7.5 Substack draft is in
`substack-phase-seven-result.txt`.

The locked-conclusion plan for provisional memory and the final synthetic
benchmark is in `../phase8_consolidation/phase8-plan.md`.
