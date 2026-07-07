# Phase 4 initial result

Command:

```bash
.venv/bin/python -m phase4_learned_gate.experiment --gate-steps 500
.venv/bin/python -m unittest phase4_learned_gate.test_experiment -q
```

Phase 4 replaces the supplied contextual gate from Phase 3 with a learned
sequence model. The learned N module reads token sequences containing optional
`REVOKE key` / `RESTORE key` controls plus a final `QUERY key`, and predicts one
of three classes:

- `no context`
- `revoked`
- `restored`

Retrieval remains algorithmic. The purpose of this phase is to test whether the
Phase 3 retention mechanism survives when contextual permission is inferred from
tokens rather than supplied directly.

## Summary over seeds 7, 17, 27, 37, 47

| method | gate class | answer | short probe preserve | long probe consolidate | short update gate | long update gate |
|---|---:|---:|---:|---:|---:|---:|
| integrated_idl | 1.0000 ± 0.0000 | 0.9914 ± 0.0101 | 0.9785 ± 0.0295 | 0.9765 ± 0.0275 | 0.0118 ± 0.0137 | 0.3951 ± 0.0105 |
| always_update | 1.0000 ± 0.0000 | 0.9426 ± 0.0418 | 0.6938 ± 0.0780 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| fixed_slow | 1.0000 ± 0.0000 | 0.9062 ± 0.0258 | 0.7906 ± 0.0538 | 0.7229 ± 0.0614 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| context_only | 1.0000 ± 0.0000 | 0.8179 ± 0.0088 | 0.6612 ± 0.0580 | 0.3675 ± 0.0313 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## Interpretation

The learned contextual gate reached perfect class accuracy on the stochastic
runtime stream. With that gate in place, the retention result matches Phase 3:

- current context behavior is correct;
- short overrides mostly do not rewrite retained policy;
- long overrides mostly do rewrite retained policy;
- always-update, fixed-slow, and context-only baselines fail the expected sides
  of the tradeoff.

This is a positive quick follow-on, but not yet a hard neural result. The gate
classifier's input format is simple, and retrieval is still algorithmic. The
result shows that replacing the supplied gate with a learned token-conditioned
gate does not break the Chevron/IDL retention loop. It does not yet show that a
full Transformer learns retrieval, contextual gating, and retained updating
jointly.
