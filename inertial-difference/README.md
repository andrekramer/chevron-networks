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

## 4. Task 1 Recovery

This PyTorch experiment tests whether retained N structure can improve external
behavior when an old task returns.

The stream has three phases:

- Task 1 is learned first until it is stable.
- Task 2 is then learned without replaying Task 1.
- Task 1 then returns.

The comparison is:

- `MLP`: ordinary online MLP.
- `ChevronSlow`: A/N chevron where N is mostly overwritten during Task 2.
- `IDLGated`: A/N chevron where N retains more Task 1 structure during Task 2.

This experiment adds an explicit N-to-A recovery constraint during Task 1
phases. That tests whether retained structure can steer fast adaptation when an
old regime returns.

Run:

```bash
.venv-torch/bin/python task1_recovery_experiment.py
```

Outputs:

- `runs_task1_recovery/probes.csv`: Task 1 and Task 2 probe accuracy through
  all three phases.
- `runs_task1_recovery/summary.csv`: recovery speed and end-state metrics.

The central diagnostic is whether IDL recovers Task 1 faster after Task 2.

Current default result over eight seeds:

- After Task 2, `IDLGated` has retained much more Task 1 in N (`0.816`) than
  `ChevronSlow` (`0.260`).
- At 10 return steps, `IDLGated` reaches about `0.601` Task 1 accuracy, versus
  about `0.436` for the MLP.
- `IDLGated` reaches 95% of its Task 1 baseline in about `26.2` return steps,
  versus `30.0` for the MLP and `33.8` for `ChevronSlow`.

This supports a stronger behavioral claim than the catastrophic-forgetting
probe alone: retained structure in N can help the system recover an old task
faster, provided N is allowed to constrain A during recovery.

## 5. Rapid Provisional Learning

This PyTorch experiment tests whether A can jump to a fast provisional
conclusion while N consolidates only repeated patterns.

The stream has three ingredients:

- a stable background task learned first,
- one-shot local exception rules,
- recurring local exception rules.

For the chevron models, each episode starts by copying N into A. A then adapts
quickly to a tiny support set. After the episode, N may consolidate toward A.

The comparison is:

- `MLP`: ordinary online MLP, where every one-shot update changes the same
  long-term weights.
- `ChevronSlow`: A/N chevron where N consolidates every episode.
- `IDLGated`: A/N chevron where N consolidates only when a rule recurs.

Run:

```bash
.venv-torch/bin/python rapid_learning_experiment.py
```

Outputs:

- `runs_rapid_learning/episodes.csv`: per-episode provisional accuracy,
  retention gate, and background accuracy.
- `runs_rapid_learning/summary_by_seed.csv`: per-seed aggregate metrics.
- `runs_rapid_learning/summary.csv`: aggregate metrics over all seeds.

The central diagnostic is whether A can adapt rapidly while N is selective
about what becomes retained structure.

Current default result over eight seeds:

- `IDLGated` still adapts rapidly on one-shot rules (`0.927` provisional
  accuracy), though not as strongly as the MLP (`0.963`).
- `IDLGated` consolidates recurring rules into N about as well as
  `ChevronSlow` (`0.696` vs `0.698`).
- `IDLGated` retains fewer one-shot rules in N than the baselines (`0.454`,
  versus `0.513` for `ChevronSlow` and `0.595` for the MLP).
- `IDLGated` has the best consolidation selectivity, measured as recurring N
  accuracy minus one-shot N accuracy (`0.242`, versus `0.185` for
  `ChevronSlow` and `0.175` for the MLP).
- `IDLGated` also preserves the background task best (`0.806`, versus `0.786`
  for `ChevronSlow` and `0.727` for the MLP).

This supports the fifth diagnostic claim: A can make fast provisional
adaptations, while N is more selective about which fast adaptations become
long-term structure.
