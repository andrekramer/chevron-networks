# Phase 5: complete learned Chevron memory network

Phase 5 removes Phase 4's algorithmic retrieval scaffold. A single trained
network now learns all of the observable computation:

- fast `A` key states produce `Q_A` and `K_A` and select a fact;
- retained `N` value states produce `V_N`;
- a learned `N` control path predicts `no context`, `revoked`, or `restored`
  for every fact;
- the answer head learns value recall and abstention jointly;
- explicit IDL dynamics decide whether the predicted contextual permission is
  temporary or becomes the retained default.

The follow-on article is in `substack-phase-five-result.md`.

The core attention operation is the ART-adjacent form proposed in the design
note:

```text
alpha = softmax(Q_A K_A^T)
g     = p(no-context) * N_retained + p(restored)
y     = sum(alpha * g * V_N) + (1 - sum(alpha * g)) * V_null
```

Training uses fresh random key/value bindings and independently randomized
retained permissions. That prevents the answer head from memorizing a global
key/value map or treating `no context` as synonymous with permission.

The online demo then freezes every neural parameter. Short and long revoke /
restore episodes affect behavior immediately, while only the IDL state may
change. It compares the same learned network under four retention rules.

Run the quick test:

```bash
.venv/bin/python -m unittest phase5_complete_network.test_experiment -q
```

Run the multi-seed demo:

```bash
.venv/bin/python -m phase5_complete_network.experiment --device auto
```

The first executed three-seed result is recorded in `initial-results.md`.

Run the robustness suite (duration, noise, scale, and IDL sensitivity):

```bash
.venv/bin/python -m phase5_complete_network.robustness --device auto
```

The executed sweep and its negative findings are recorded in
`robustness-results.md`.

Compare hard direction reset with two-trace and signed-hysteresis persistence:

```bash
.venv/bin/python -m phase5_complete_network.direction_comparison --device auto
```

See `direction-comparison-results.md` for the executed comparison and
recommendation.

For a quick smoke run:

```bash
.venv/bin/python -m phase5_complete_network.experiment --seeds 7 --steps 350 --device cpu
```

## Why this is not yet ART reset/search

`Q_A, K_A, V_N` is already the useful ART-like part: present evidence selects
a retained template/value. Full ART adds reset and category search after
mismatch. On this task there is exactly one correct key/value binding, so
searching after a permission veto would select an irrelevant fact rather than
abstain. The clean next ablation is therefore resonance/vigilance on a task
with several plausible templates or genuinely novel categories. Adding reset
here would make a flashier mechanism but a less meaningful experiment.

## Interpretation boundary

This is a complete small neural system, not a full language Transformer. Q/K
retrieval, N values, contextual gating, and answer production are learned
jointly; the IDL consolidation equation remains an explicit architectural
rule. The experiment tests whether learned neural signals can drive the
stability/plasticity mechanism. It does not show that gradient descent will
discover IDL on its own or that the method improves standard language-model
benchmarks.
