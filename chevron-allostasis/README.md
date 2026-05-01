# Chevron Allostasis

Standalone first experiment for allostatic A/N chevron learning on Split MNIST.

The initial implementation intentionally starts with the direct chevron path only:

- paired A/N hidden state with local 2x2 chevron coupling
- sequential Split MNIST tasks: `0/1`, `2/3`, `4/5`, `6/7`, `8/9`
- wake training with asymmetric plasticity for A-facing and N-facing channels
- optional uniform replay buffer for consolidation
- accuracy matrix, forgetting score, A/N disagreement, and coupling norms

## Setup

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python allostatic_chevron.py --self-test --max-train-per-task 256 --max-test-per-task 256 --epochs 1
```

## First Real Run

```bash
python allostatic_chevron.py --epochs 3 --width 128 --batch-size 128 --buffer-size 0
```

`chevron` is the default model. To run the scalar baseline:

```bash
python allostatic_chevron.py --model mlp --epochs 3 --width 128 --batch-size 128 --buffer-size 500 --consolidation-epochs 1
```

Add uniform replay/consolidation:

```bash
python allostatic_chevron.py --epochs 3 --width 128 --batch-size 128 --buffer-size 500 --consolidation-epochs 1
```

Compare replay policies:

```bash
python allostatic_chevron.py --readout a_only --buffer-size 500 --consolidation-epochs 1 --replay-policy uniform
python allostatic_chevron.py --readout a_only --buffer-size 500 --consolidation-epochs 1 --replay-policy disagreement
python allostatic_chevron.py --readout a_only --buffer-size 500 --consolidation-epochs 1 --replay-policy loss_disagreement
```

Current replay-policy result: `disagreement` is the best first tension-guided policy. `loss_disagreement` is available for tuning, but raw loss is not a reliable priority signal in the first sweeps.

Compare dream schedules:

```bash
python allostatic_chevron.py --readout a_only --buffer-size 500 --consolidation-epochs 1 --replay-policy disagreement --dream-schedule post_task
python allostatic_chevron.py --readout a_only --buffer-size 500 --consolidation-epochs 1 --replay-policy disagreement --dream-schedule after_epoch
```

Current dream-schedule result: fixed `after_epoch` dreaming is not better than `post_task` consolidation in the first sweep. It improves current-task accuracy but worsens old-task retention, suggesting the next test should gate dream by A/N tension rather than by clock time.

First tension-gated result: absolute wake-tension thresholds were brittle. Low thresholds behaved like `after_epoch`; high thresholds skipped too much dreaming and regressed toward no-replay forgetting. The next gate should use relative tension change or persistent tension, not a fixed global threshold.

First persistent-tension result: epoch-average persistence was too strict and skipped useful dreams, regressing toward no-replay forgetting. The next gating test should track high-tension replay examples or use a post-task fallback, rather than relying only on global wake-epoch averages.
