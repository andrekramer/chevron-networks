# Phase 8: provisional memory and consolidation

Phase 8 tests whether unassented evidence should enter a small provisional
memory before it is allowed to alter retained category memory.

The first implementation keeps unused mass and novelty distinct:

```text
q = 1 - sum(alpha * r)
nu = 1 - max(r over top-A candidates)
u = q * nu
```

Only `u` enters a fixed-capacity candidate bank. Coherent observations update a
candidate and its persistence trace. Consolidation is allowed only after
minimum persistence, support, and distinctness requirements are all met.

Run the deterministic mechanism tests:

```bash
.venv/bin/python -m unittest phase8_consolidation.test_provisional_memory -q
```

The implementation calls the persistence trace MTM-inspired, not an
implementation of Grossberg's MTM. It accumulates evidence for consolidation;
canonical habituative transmitter gates generally weaken with recent use and
serve a different reset/search function.

The next milestone integrates this bank with the frozen Phase 7 Chevron and
joint-controller write policies, then runs the development-seed comparison
specified in `phase8-plan.md`.

Run the immediate-versus-quarantine comparison:

```bash
.venv/bin/python -m phase8_consolidation.experiment
```

Run the first persistence-parameter screen on development seeds:

```bash
.venv/bin/python -m phase8_consolidation.development_sweep
```

The sweep uses a predeclared stability-plasticity objective rather than
selecting parameters by final accuracy.

The first screen selected `beta=0.8`, an initial `tau_P=0.25`, minimum support `5`,
minimum eligible mass `0.10`, and distinct mismatch `0.04`. These remain
development parameters rather than confirmation results.

The resulting three-seed comparison is discussed in
`development-results.md`. Machine-readable per-stream outputs are in
`development-results.json`, and the full screen is in
`development-sweep.json`.

Run the pre-lock robustness matrix on five development seeds:

```bash
.venv/bin/python -m phase8_consolidation.robustness
```

The executed matrix and pre-lock decision are recorded in
`robustness-results.md`. The selected shared configuration did not pass the
joint-controller interleaved-acquisition criterion. The focused correction and
rerun are recorded in `revision-results.md`; the passing configuration is now
frozen in `locked-config.json`, with `tau_P=0.20` and minimum support still `5`.

Run the focused revision check:

```bash
.venv/bin/python -m phase8_consolidation.revision_check
```

Run the locked confirmation protocol:

```bash
.venv/bin/python -m phase8_consolidation.confirmation
```

The exact new seeds, conditions, comparisons, and hypotheses are declared in
`confirmation-protocol.md`.

The completed locked run is reported in `confirmation-results.md`, with raw
rows and paired bootstrap intervals in `confirmation-results.json`. H3 and H4
passed; the strict cross-method H1 and H2 hypotheses failed. Chevron quarantine
itself replicated the intended temporal boundary and near-category advantage.

The final plain-text Substack article for the Chevron Attention series is in
`substack-chevron-attention-final.txt`.
