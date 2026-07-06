# Phase two: persistent difference and retained change

This directory contains a separate minimal experiment for Inertial Difference
Learning. It does not depend on the phase-one attention implementation.

## Hypothesis

A fast state `A` should track current evidence. A slower retained state `N`
should resist a brief contradiction but adapt when the same difference persists.

IDL uses:

```text
D_t   = RMS(A_t - N_t)
P_t   = beta P_(t-1) + (1 - beta) D_t
rho_t = sigmoid(k (P_t - theta))
N_t   = N_(t-1) + eta_N rho_t (A_t - N_(t-1))
```

The `idl_scaled` variant replaces raw difference in the persistence trace with a
bounded excess above an online estimate of ordinary discrepancy. The estimate is
updated only while difference remains inside a configurable noise margin, so a
persistent anomaly cannot redefine itself as normal before retention opens.

## Protocol

An online linear predictor starts from a learned base mapping and passes through
four phases:

1. stable base mapping, long enough for every retained baseline to converge;
2. brief contradictory mapping;
3. return to the base mapping;
4. sustained change to the contradictory mapping.

Every method receives identical samples and uses the same fast learner. The
comparison changes only the retained-state rule:

- `idl`: persistence-gated slow update;
- `idl_scaled`: scale-aware persistence relative to learned ordinary discrepancy;
- `always_slow`: the same maximum slow rate, applied at every step;
- `fixed_slow_low`: a lower fixed rate that reduces temporary drift but must
  pay for that protection during sustained adaptation;
- `fast_only`: no distinct retained state.

Primary metrics are retained-state drift during the brief contradiction,
recovery error, steps to adapt after sustained change, final sustained error,
and the gate response in both change periods.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m phase2_idl.experiment
python -m phase2_idl.sweep
python -m unittest phase2_idl.test_experiment -v
```

The default command runs five seeds. All schedule and IDL parameters are exposed
as command-line options; use `python -m phase2_idl.experiment --help` for the
complete list.

## Success criterion

IDL must move `N` less than the always-updating slow baseline during the brief
contradiction, while still opening its gate and reaching low retained-state error
after the sustained change. Reduced temporary drift alone is not sufficient: a
retained state that never adapts trivially achieves that result.

The checked default parameters and preliminary five-seed output are recorded in
[`initial-results.md`](initial-results.md).

The one-factor-at-a-time sensitivity study is recorded in
[`sweep-results.md`](sweep-results.md).
