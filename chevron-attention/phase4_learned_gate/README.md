# Phase 4: learned contextual gate

Phase 4 replaces the supplied contextual gate from Phase 3 with a small learned
sequence model.

- A retrieval remains algorithmic.
- A learned contextual N module reads control/query tokens and predicts one of:
  `no context`, `revoked`, or `restored`.
- The predicted contextual gate drives the same IDL retained-policy update used
  in Phase 3.

Run:

```bash
.venv/bin/python -m phase4_learned_gate.experiment
.venv/bin/python -m unittest phase4_learned_gate.test_experiment -q
```

This is intentionally not a full Transformer experiment. It isolates whether
the Phase 3 retention result survives replacing the oracle contextual gate with
a learned token-conditioned gate.
