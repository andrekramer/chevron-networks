# Phase 3: integrated Chevron/IDL experiment

This experiment combines the two mechanisms tested separately in Phase 1 and
Phase 2.

- A retrieves the queried fact value.
- Contextual N gates the current answer immediately.
- Retained N changes only when a contextual override persists.

The schedule is a full permission cycle:

1. active fact use with no control context;
2. a brief revoke burst;
3. no context again, where the original active policy should survive;
4. a sustained revoke, where retained N should consolidate revocation;
5. no context again, where the revocation should persist;
6. a sustained restore, where retained N should consolidate restoration;
7. final no-context active behavior.

Run:

```bash
../.venv/bin/python -m phase3_integrated.experiment
```

From the repository root:

```bash
.venv/bin/python -m phase3_integrated.experiment
.venv/bin/python -m phase3_integrated.sweep
.venv/bin/python -m phase3_integrated.stochastic
.venv/bin/python -m unittest phase3_integrated.test_experiment -q
```

Methods:

- `integrated_idl`: immediate contextual behavior, persistence-gated retained
  updates.
- `always_update`: retained policy follows every contextual override.
- `fixed_slow`: retained policy follows every contextual override slowly.
- `context_only`: contextual behavior changes immediately, but retained policy
  never changes.

The intended positive result is not higher raw task accuracy. It is a tradeoff:
temporary controls should affect current behavior without overwriting retained
policy, while persistent controls should eventually rewrite the retained policy.

See:

- `initial-results.md`
- `sweep-results.md`
- `stochastic-results.md`
