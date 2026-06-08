# Inertial Difference Learning Test

This directory contains a small, dependency-free test harness for the idea in
`idl.txt`.

The stream is an online binary classification task with three pressures:

- a stable pre-shift concept,
- short contradictory bursts before the shift,
- a persistent post-shift concept.

The comparison is:

- `MLP`: ordinary online MLP.
- `ChevronSlow`: A/N chevron with fast A and slow N, where N always updates.
- `IDLGated`: A/N chevron where persistent `A - N` difference gates retention
  into N.

Run:

```bash
.venv/bin/python run_idl_experiment.py
```

Outputs:

- `runs/trace.csv`: per-step predictions and internal signals.
- `runs/summary.csv`: aggregate windows for transient bursts, pre-shift
  stability, shift adaptation, and post-shift recovery.

The central diagnostic is whether the gated model keeps `rho` lower during
brief contradictions and raises it during persistent shift pressure.
