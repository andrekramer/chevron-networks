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
