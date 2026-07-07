# Phase 3 initial result

Command:

```bash
.venv/bin/python -m phase3_integrated.experiment
.venv/bin/python -m unittest phase3_integrated.test_experiment -q
```

Environment note: PyTorch runs in the project virtual environment. It emits the
same NumPy warning seen in earlier phases because NumPy is not installed, but
the experiment does not require NumPy.

## Summary over seeds 7, 17, 27, 37, 47

The current seed only changes the key/value assignment. Because the control
schedule and update dynamics are deterministic, metric variance is zero in this
first implementation.

| method | retrieval | answer | transient drift | recovery active | post-revoke retained | final active | revoke steps | restore steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| integrated_idl | 1.000 | 1.000 | 0.0018 | 1.000 | 1.000 | 1.000 | 32 | 12 |
| always_update | 1.000 | 0.917 | 0.5656 | 0.000 | 1.000 | 1.000 | 1 | 9 |
| fixed_slow | 1.000 | 1.000 | 0.1137 | 1.000 | 1.000 | 1.000 | 48 | 49 |
| context_only | 1.000 | 0.917 | 0.0000 | 1.000 | 0.000 | 1.000 | 181 | 1 |

## Interpretation

The integrated condition shows the intended full cycle:

- A retrieval remains intact: retrieval accuracy is 1.000.
- Contextual N controls immediate behavior: brief revoke, sustained revoke, and
  sustained restore all have current-behavior accuracy of 1.000.
- Retained N barely moves during the brief revoke: retained drift is 0.0018.
- Retained N changes after persistent revoke: no-context post-revoke accuracy is
  1.000.
- Retained N changes back after persistent restore: final no-context active
  accuracy is 1.000.

The baselines show different missing pieces:

- `always_update` consolidates immediately and therefore overwrites retained
  policy during the brief revoke; recovery active accuracy is 0.000.
- `context_only` handles immediate behavior but never changes retained policy;
  post-revoke retained accuracy is 0.000.
- `fixed_slow` works on this exact schedule, but it does so with a fixed
  timescale rather than a persistence-sensitive gate. It drifts more than IDL
  during the transient and consolidates more slowly in both persistent phases.

This is a positive integration check, not yet a strong benchmark. The next
useful step is a sweep over transient duration, sustained duration, fixed-slow
rates, and stochastic query/control schedules.
