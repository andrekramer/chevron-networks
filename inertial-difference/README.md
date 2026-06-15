# Inertial Difference Learning Experiments

This directory contains small test harnesses for the idea in `../idl.txt`.

## 1. Signal Detection

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
.venv/bin/python signal_idl_experiment.py
```

Outputs:

- `runs/trace.csv`: per-step predictions and internal signals.
- `runs/summary.csv`: aggregate windows for transient bursts, pre-shift
  stability, shift adaptation, and post-shift recovery.

The central diagnostic is whether the gated model keeps `rho` lower during
brief contradictions and raises it during persistent shift pressure.

## 2. Temporary Difference

This PyTorch experiment tests the next question after signal detection: whether
IDL can resist temporary difference without becoming unable to retain persistent
difference.

The stream is an online binary classification task with four pressures:

- a stable concept,
- a short contradictory concept that should be treated as temporary,
- a recovery period back to the original stable concept,
- a later persistent shift to the contradictory concept.

The comparison is:

- `MLP`: ordinary online MLP.
- `ChevronSlow`: A/N chevron with fast A and slow N, where N always updates.
- `IDLGated`: A/N chevron where excess persistent `A - N` difference gates
  retention into N.

Run:

```bash
.venv-torch/bin/python temporary_difference_experiment.py
```

Outputs:

- `runs_temporary_difference/trace.csv`: per-step batch metrics and internal
  signals.
- `runs_temporary_difference/summary.csv`: aggregate diagnostic windows.

The central diagnostic is whether the gated model moves N less during the brief
temporary difference, but opens retention during the later persistent shift.

Current default result over six seeds:

- During the temporary difference, `IDLGated` moves N about half as much as
  `ChevronSlow` (`NmoveT` about `0.0241` vs `0.0502`).
- During the persistent shift, `IDLGated` opens retention strongly (`rho` about
  `0.899` in the shift adaptation window).

This supports the second diagnostic claim: brief difference is partly resisted,
while sustained difference is treated as a candidate for retention.

## 3. Catastrophic Forgetting

This PyTorch experiment tests whether IDL can reduce catastrophic forgetting in
a sequential learning setting.

The stream has two tasks:

- Task 1 is learned first until it is stable.
- Task 2 is then learned without replaying Task 1.

The comparison is:

- `MLP`: ordinary online MLP.
- `ChevronSlow`: A/N chevron where both A and N continue updating during Task 2.
- `IDLGated`: A/N chevron where A adapts to Task 2 while N is protected by an
  IDL retention gate.

Run:

```bash
.venv-torch/bin/python catastrophic_forgetting_experiment.py
```

Outputs:

- `runs_catastrophic_forgetting/probes.csv`: Task 1 and Task 2 probe accuracy
  through training.
- `runs_catastrophic_forgetting/summary.csv`: end-of-Task-A, end-of-Task-B,
  and forgetting summary metrics.

The central diagnostic is whether Task 2 can be learned while the retained N
channel still preserves Task 1 better than the baselines.

Current default result over eight seeds:

- After Task 1, all models are near `0.99` accuracy on Task 1.
- After Task 2, the ordinary MLP has fallen to `0.163` on Task 1 while reaching
  `0.950` on Task 2.
- After Task 2, `ChevronSlow` has also largely overwritten N: N retains only
  `0.192` Task 1 accuracy.
- After Task 2, `IDLGated` reaches `0.958` combined Task 2 accuracy while its N
  channel still retains `0.784` Task 1 accuracy.

This supports the third diagnostic claim in a limited but useful form: IDL does
not eliminate forgetting, but it can create a window where the adaptive channel
learns the new task while the retained channel still preserves much more of the
old task.
