# Phase 6: ART-inspired category learning

Phase 6 gives vigilance and search a task on which they are necessary. The
stream contains:

1. three recurring base categories;
2. eight coherent but brief categories;
3. a read-only probe of the original categories;
4. one persistent new category;
5. a final read-only probe of old and new knowledge.

Every method predicts before seeing the label. Labels are revealed only on
learning phases; probes do not modify state.

The initial Chevron mechanism is deliberately small:

```text
scores = Q(A_current) K(A_slot)^T
candidate slots are considered in score order

C = (A_current - N_slot) / (A_current + N_slot + epsilon)
G = 2 sqrt(p(1-p)),  p = (A_current + epsilon/2) / (A_current + N_slot + epsilon)

low mismatch + high complementarity -> resonate and update
mismatch                           -> veto and search next slot
no resonant slot                   -> update fast candidate A
persistent coherent candidate      -> create retained A/N category
```

The complementarity gate is not itself a mismatch detector: it is maximal at
`A == N`. Phase 6 therefore uses absolute normalized contrast as vigilance and
uses complementarity as a resonance condition.

## Comparisons

- `standard_attention`: ordinary softmax Q/K/V retrieval with immediate
  supervised writes and LRU replacement. It receives nine feature-vector
  slots, matching the approximate feature-state budget of four two-vector
  Chevron slots plus one fast candidate.
- `persistent_attention`: the strongest structural ablation. It uses ordinary
  single-template attention with the same vigilance, search, fast candidate,
  and persistence-gated creation rule. It receives eight retained templates
  plus one candidate, matching the same feature-state budget. A tie here means
  the experiment supports persistent category allocation, not a specifically
  Chevron A/N advantage.
- `online_mlp`: a plain online MLP with the complete output vocabulary known in
  advance. This is a favorable fixed-vocabulary control, not a memory-matched
  architecture.
- `chevron_art`: A-key attention, N-template vigilance, reset/search, and
  persistence-gated category creation.

This first experiment isolates category allocation and retention. It does not
yet train Q/K projections or claim a general continual-learning advantage.

Run the tests:

```bash
.venv/bin/python -m unittest phase6_category_learning.test_experiment -q
```

Run five seeds:

```bash
.venv/bin/python -m phase6_category_learning.experiment
```

## Phase 6.1: recurring category drift

The first category-allocation task is solved equally well by Chevron and
single-template persistent attention. `drift_comparison.py` therefore tests a
case where the current category surface moves temporarily, returns, and later
moves persistently. It compares fast, slow, and persistence-gated single
templates with Chevron A/N traces and standard attention over the same two
traces.

```bash
.venv/bin/python -m phase6_category_learning.drift_comparison
```

The executed five-seed comparison is recorded in `drift-results.md`. Its main
negative finding is that soft-contrast Chevron ties ordinary attention over the
same A/N traces, while hard veto/search adapts more slowly.

## Phase 6.2: contextual ambiguity

`ambiguity_comparison.py` gives ordered search a legitimate alternative
category. Bottom-up decoys outrank the target on A similarity, but only the
target N template satisfies the query context. The comparison increases the
number of correlated decoys and tests ordinary A-only and joint softmax
attention, joint top-1 retrieval, complementarity-only gating, differentiable
vigilance, and hard reset/search.

```bash
.venv/bin/python -m phase6_category_learning.ambiguity_comparison
```

See `ambiguity-results.md` for the executed comparison. Hard search succeeds
after as many as 31 resets, but joint top-1 retrieval, sharp differentiable
vigilance, and masked standard attention tie it. Complementarity by itself is
not a viable vigilance signal.

## Final robustness suite

`robustness.py` runs the category-creation, drift, and ambiguity mechanisms over
20 paired seeds and one-factor difficulty sweeps using frozen default
hyperparameters.

```bash
.venv/bin/python -m phase6_category_learning.robustness
```

Individual sections can be run with `--section category`, `--section drift`, or
`--section ambiguity`.

The executed 20-seed results and Phase 6 conclusion are in
`phase6-final-results.md`.
